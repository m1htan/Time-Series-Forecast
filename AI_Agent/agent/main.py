from typing import TypedDict
import pandas as pd
from langgraph.graph import StateGraph
from AI_Agent.agent.tools import crawl_data_tool, save_data_tool, fill_missing_dates_tool
from langchain_core.runnables import RunnableLambda

# 1. Định nghĩa State schema
class StockState(TypedDict):
    stock_data: pd.DataFrame
    preprocessed_data: pd.DataFrame

# 2. Khởi tạo StateGraph có schema
workflow = StateGraph(StockState)

# 3. Thêm các node xử lý
workflow.add_node("crawl_data", crawl_data_tool)
workflow.add_node("fill_missing", fill_missing_dates_tool)
workflow.add_node("save_data_node", save_data_tool)

# 4. Kiểm thử trung gian
@RunnableLambda
def test_output(state):
    data = state.get("stock_data")

    if isinstance(data, dict):
        for ticker, df in data.items():
            print(f"\n=== {ticker} ===\n{df.head()}")
    elif isinstance(data, pd.DataFrame):
        print("5 dòng đầu tiên:\n", data.head())
    else:
        print("Dữ liệu không hợp lệ.")

    return state

workflow.add_node("test_node", test_output)

# 5. Kết nối các bước
workflow.set_entry_point("crawl_data")
workflow.add_edge("crawl_data", "fill_missing")
workflow.add_edge("fill_missing", "test_node")
workflow.add_edge("test_node", "save_data_node")
workflow.set_finish_point("save_data_node")

# 6. Compile graph
graph = workflow.compile()

# 7. Chạy graph
if __name__ == "__main__":
    result = graph.invoke({})