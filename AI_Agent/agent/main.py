from typing import TypedDict
import pandas as pd
from langgraph.graph import StateGraph
from langchain_core.runnables import RunnableLambda

from AI_Agent.tools.check_preprocessing import check_preprocessing_tool
from AI_Agent.tools.fill_missing_data import fill_missing_dates_tool
from AI_Agent.tools.main_crawl_data import crawl_data_tool
from AI_Agent.tools.save_data_to_csv_daily import save_data_tool
from AI_Agent.tools.checking_logs import log_workflow_step_tool

# 1. Định nghĩa State schema ------------------------------------------------------------------------------------
class StockState(TypedDict):
    stock_data: pd.DataFrame
    preprocessed_data: pd.DataFrame

# 2. Khởi tạo StateGraph có schema ------------------------------------------------------------------------------
workflow = StateGraph(StockState)

# 3. Kiểm thử trung gian ----------------------------------------------------------------------------------------
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

# 4. Thêm các node xử lý ----------------------------------------------------------------------------------------
workflow.add_node("crawl_data", crawl_data_tool)
workflow.add_node("log_after_crawl", lambda state: log_workflow_step_tool.invoke("Đã crawl dữ liệu"))

workflow.add_node("fill_missing", fill_missing_dates_tool)
workflow.add_node("log_after_fill", lambda state: log_workflow_step_tool.invoke("Đã fill dữ liệu bị thiếu"))

workflow.add_node("test_node", test_output)

workflow.add_node("check_preprocessing_tool", check_preprocessing_tool)
workflow.add_node("log_after_check", lambda state: log_workflow_step_tool.invoke("Đã kiểm tra preprocessing"))

workflow.add_node("save_data_node", save_data_tool)

# 5. Kết nối các bước -------------------------------------------------------------------------------------------
workflow.set_entry_point("crawl_data")

workflow.add_edge("crawl_data", "log_after_crawl")
workflow.add_edge("log_after_crawl", "fill_missing")

workflow.add_edge("fill_missing", "log_after_fill")
workflow.add_edge("log_after_fill", "test_node")

workflow.add_edge("test_node", "check_preprocessing_tool")
workflow.add_edge("check_preprocessing_tool", "log_after_check")

workflow.add_edge("log_after_check", "save_data_node")
workflow.set_finish_point("save_data_node")

# 6. Compile graph ----------------------------------------------------------------------------------------------
graph = workflow.compile()

# 7. Chạy graph -------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    result = graph.invoke({})