from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import TypedDict, List, Optional, Dict
from pydantic import Field
import matplotlib.pyplot as plt
import os
import re
import smtplib
import pandas as pd
import time
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from dotenv import load_dotenv

# Load biến môi trường
load_dotenv(dotenv_path='/Users/minhtan/Documents/GitHub/Time_Series_Forecast/AI_Agent/config/config.env')
SENDER_EMAIL = os.getenv("SENDER_EMAIL")
SENDER_PASSWORD = os.getenv("SENDER_PASSWORD")
RECIPIENT_EMAIL = os.getenv("RECIPIENT_EMAIL")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Khởi tạo LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-pro",
    temperature=0.7,
    google_api_key=GOOGLE_API_KEY
)

# Tool gửi email
@tool
def send_report_email(
    subject: str = Field(...),
    html_body: str = Field(...),
    image_paths: List[str] = Field(default_factory=list)
) -> str:
    """
    Gửi email báo cáo chiến lược đầu tư có kèm biểu đồ.
    """
    try:
        msg = MIMEMultipart('related')
        msg['Subject'] = subject
        msg['From'] = SENDER_EMAIL
        msg['To'] = RECIPIENT_EMAIL

        msg.attach(MIMEText(html_body, 'html'))

        for i, path in enumerate(image_paths):
            if not os.path.exists(path):
                continue
            with open(path, 'rb') as f:
                img = MIMEImage(f.read())
                img.add_header('Content-ID', f'<chart{i}>')
                msg.attach(img)

        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.set_debuglevel(1)
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
            server.send_message(msg)

        return "Email sent successfully."
    except Exception as e:
        return f"Failed to send email: {e}"

class ReportState(TypedDict):
    initial_strategy_text: str
    recipient_email: str
    chart_paths: Optional[List[str]]
    messages: List[BaseMessage]


def chart_generator_node(state: ReportState) -> Dict:
    strategy = state["initial_strategy_text"]
    allocations = re.findall(r"\*\*([A-Z]+)\*\*.*?\$\s*([\d,]+\.?\d*)", strategy)
    if not allocations:
        return {"chart_paths": []}
    labels = [item[0] for item in allocations]
    sizes = [float(item[1].replace(',', '')) for item in allocations]
    fig, ax = plt.subplots()
    ax.pie(sizes, labels=labels, autopct='%1.1f%%')
    ax.axis('equal')
    chart_path = f"allocation_pie_chart_{int(time.time())}.png"
    plt.savefig(chart_path)
    plt.close(fig)
    return {"chart_paths": [chart_path]}


def supervisor_node(state: ReportState) -> Dict:
    llm_with_tools = llm.bind_tools([send_report_email])
    prompt = ChatPromptTemplate.from_messages([
        ("system", """Bạn là trợ lý báo cáo tài chính. Soạn nội dung HTML, nhúng hình ảnh cid và gọi đúng công cụ `send_report_email` với thông tin đã cung cấp."""),
        *state["messages"],
        ("user", f"""- Email: {state['recipient_email']}
- Nội dung: {state['initial_strategy_text']}
- Ảnh: {state['chart_paths']}""")
    ])
    chain = prompt | llm_with_tools
    result = chain.invoke(state)
    return {"messages": state["messages"] + [result]}


def should_continue(state: ReportState) -> str:
    if state["messages"][-1].tool_calls:
        return "call_tool"
    return "end"


@tool
def send_report_tool(recipient_email: str = Field(...)) -> str:
    """
    Gửi email báo cáo đầu tư từ chiến lược được lưu trong investment_decision.csv
    """
    # Đọc file CSV để lấy nội dung chiến lược
    csv_path = "/Users/minhtan/Documents/GitHub/Time_Series_Forecast/AI_Agent/output/investment_results/investment_decision.csv"
    if not os.path.exists(csv_path):
        return "[ERROR] File investment_decision.csv không tồn tại."

    df = pd.read_csv(csv_path)
    if "strategy" not in df.columns or df.empty:
        return "[ERROR] Không có nội dung chiến lược để gửi."

    strategy_text = str(df["strategy"].iloc[-1]).strip()
    if not strategy_text:
        return "[ERROR] Chiến lược rỗng."

    # Xây dựng workflow LangGraph
    report_graph = StateGraph(ReportState)
    report_graph.add_node("chart_generator", chart_generator_node)
    report_graph.add_node("supervisor", supervisor_node)
    report_graph.add_node("call_tool", ToolNode([send_report_email]))

    report_graph.set_entry_point("chart_generator")
    report_graph.add_edge("chart_generator", "supervisor")
    report_graph.add_conditional_edges("supervisor", should_continue, {
        "call_tool": "call_tool",
        "end": END
    })
    report_graph.add_edge("call_tool", END)

    report_app = report_graph.compile()

    result = report_app.invoke({
        "initial_strategy_text": strategy_text,
        "recipient_email": recipient_email,
        "messages": [HumanMessage(content="Bắt đầu gửi báo cáo.")]
    })

    return "Đã gửi email báo cáo thành công!"
