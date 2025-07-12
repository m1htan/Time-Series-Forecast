from typing import Dict

import pandas as pd
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_core.pydantic_v1 import BaseModel, Field
from langgraph.graph import StateGraph, END
from langchain_core.runnables import Runnable
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
import time
import os

load_dotenv(dotenv_path='/Users/minhtan/Documents/GitHub/Time_Series_Forecast/AI_Agent/config/config.env')

# Truy cập các biến môi trường
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# 1. Định nghĩa State
class InvestmentState(BaseModel):
    """
    Đại diện cho trạng thái của đồ thị tư vấn đầu tư.
    Attributes:
        initial_capital: Số vốn ban đầu do người dùng cung cấp.
        forecast_data: Dữ liệu dự báo giá của các cổ phiếu.
        analysis: Kết quả phân tích cơ hội từ analyst_agent.
        strategy: Chiến lược phân bổ vốn cuối cùng từ strategy_agent.
        messages: Lịch sử các thông điệp trao đổi trong quá trình.
    """
    initial_capital: float
    forecast_data: dict
    analysis: str = ""
    strategy: str = ""
    initial_capital: float

# 2. Khởi tạo Gemini LLM
llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0.7, google_api_key=os.environ["GOOGLE_API_KEY"])

# 3. Agent phân tích
def analyst_agent(state: InvestmentState):
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         """Bạn là một nhà phân tích tài chính định lượng chuyên nghiệp.
Nhiệm vụ của bạn là phân tích dữ liệu dự báo giá cổ phiếu trong 5 ngày tới.
Với mỗi cổ phiếu, hãy thực hiện các việc sau:
1.  Tìm ra ngày mua vào có giá thấp nhất và ngày bán ra có giá cao nhất (ngày bán phải sau ngày mua).
2.  Tính toán lợi nhuận tiềm năng trên mỗi cổ phiếu nếu thực hiện giao dịch này.
3.  Tính toán Tỷ suất sinh lời (ROI) cho mỗi giao dịch (Lợi nhuận / Giá mua).
4.  Đưa ra nhận xét ngắn gọn về mức độ hấp dẫn của mỗi cổ phiếu.
5.  Trình bày kết quả một cách rõ ràng và có cấu trúc. Nếu một cổ phiếu không có cơ hội sinh lời (giá chỉ đi xuống), hãy nêu rõ điều đó.
"""),
        ("human",
         """Dưới đây là dữ liệu dự báo giá và thông tin tôi có:
- Dữ liệu dự báo giá 5 ngày tới: {forecast_data}
Vui lòng thực hiện phân tích của bạn.""")
    ])

    chain = prompt | llm
    forecast_data = state.forecast_data
    response = chain.invoke({"forecast_data": str(forecast_data)})
    return {"analysis": response.content, "messages": [response]}

# 4. Agent chiến lược
def strategy_agent(state: InvestmentState):
    time.sleep(60)
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         """Bạn là một nhà quản trị vốn tài chính dày dạn kinh nghiệm.
Bạn vừa nhận được một bản phân tích về cơ hội đầu tư từ đội ngũ của mình.
Nhiệm vụ của bạn là dựa vào bản phân tích này và số vốn được cung cấp để đưa ra một chiến lược phân bổ vốn chi tiết và tối ưu.
Yêu cầu:
1.  Đọc kỹ bản phân tích để hiểu các cơ hội có ROI cao nhất.
2.  Dựa trên số vốn ban đầu, quyết định phân bổ bao nhiêu tiền vào mỗi cơ hội hấp dẫn. Cân nhắc cả việc đa dạng hóa danh mục.
3.  Đưa ra kế hoạch hành động cụ thể:
    - Mua cổ phiếu nào?
    - Mua vào ngày nào?
    - Mua bao nhiêu cổ phiếu (tính toán dựa trên giá mua và số vốn phân bổ)?
    - Tổng số tiền đầu tư là bao nhiêu?
    - Số vốn còn lại là bao nhiêu?
4.  Tổng hợp lại lợi nhuận dự kiến và giá trị danh mục cuối cùng sau 5 ngày nếu chiến lược thành công.
5.  Trình bày chiến lược một cách thuyết phục, chuyên nghiệp và dễ hiểu.
"""),
        ("human",
         """Đây là thông tin tôi có:
- Số vốn ban đầu: ${initial_capital:,.2f}
- Bản phân tích cơ hội đầu tư:
{analysis}

Hãy đưa ra chiến lược phân bổ vốn của bạn.""")
    ])

    chain = prompt | llm
    response = chain.invoke({
        "initial_capital": state.initial_capital,
        "analysis": state.analysis
    })
    return {"strategy": response.content, "messages": [response]}

# 5. Tạo LangGraph Workflow
def build_investment_graph() -> Runnable:
    workflow = StateGraph(InvestmentState)
    workflow.add_node("analyst", analyst_agent)
    workflow.add_node("strategist", strategy_agent)
    workflow.set_entry_point("analyst")
    workflow.add_edge("analyst", "strategist")
    workflow.set_finish_point("strategist")
    return workflow.compile()


def investment_decision_agent(state: Dict) -> Dict:
    """
    Agent đưa ra chiến lược đầu tư, sử dụng Gemini để phân tích và lập kế hoạch đầu tư.
    Nhận state gồm 'initial_capital' và nội suy forecast_data từ thư mục output.
    """
    initial_capital = state.get("initial_capital")
    if initial_capital is None:
        raise ValueError("Thiếu initial_capital trong state.")

    # Load dữ liệu dự báo từ baseline_output
    forecast_data = {}
    folder_path = "/Users/minhtan/Documents/GitHub/Time_Series_Forecast/AI_Agent/output/baseline_output"
    for filename in os.listdir(folder_path):
        if filename.endswith("_baseline.csv"):
            ticker = filename.split("_")[0]
            df = pd.read_csv(os.path.join(folder_path, filename))
            if "Forecast" in df.columns:
                forecast_values = df["Forecast"].tolist()
            elif "Close" in df.columns:
                forecast_values = df["Close"].tolist()
            else:
                continue
            forecast_data[ticker] = {f"Day {i+1}": round(v, 2) for i, v in enumerate(forecast_values)}

    # Build workflow
    graph = build_investment_graph()
    inputs = {
        "initial_capital": initial_capital,
        "forecast_data": forecast_data,
        "messages": [HumanMessage(content="Hãy giúp tôi đưa ra chiến lược đầu tư.")]
    }

    final_state = graph.invoke(inputs)

    analysis = str(final_state.get("analysis", ""))
    strategy = str(final_state.get("strategy", ""))

    # Ghi ra file CSV
    os.makedirs("/Users/minhtan/Documents/GitHub/Time_Series_Forecast/AI_Agent/output/investment_results", exist_ok=True)
    df_result = pd.DataFrame([{
        "initial_capital": initial_capital,
        "analysis": analysis,
        "strategy": strategy
    }])
    df_result.to_csv("/Users/minhtan/Documents/GitHub/Time_Series_Forecast/AI_Agent/output/investment_results/investment_decision.csv", index=False)
    print("[INFO] Đã lưu kết quả vào /Users/minhtan/Documents/GitHub/Time_Series_Forecast/AI_Agent/output/investment_results/investment_decision.csv")

    analysis = final_state.get("analysis")
    strategy = final_state.get("strategy")

    print("Final investment graph output:", final_state)

    # Bắt buộc convert sang str nếu không chắc chắn
    if not isinstance(analysis, str):
        analysis = str(analysis)

    if not isinstance(strategy, str):
        strategy = str(strategy)

    return {
        "analysis": analysis,
        "strategy": strategy
    }
