from typing import TypedDict
import pandas as pd
from langgraph.graph import StateGraph
from langchain_core.runnables import RunnableLambda

from AI_Agent.tools.calculate_baseline import calculate_baseline_tool
from AI_Agent.tools.check_preprocessing import check_preprocessing_tool
from AI_Agent.tools.eda import eda_tool
from AI_Agent.tools.fill_missing_data import fill_missing_dates_tool
from AI_Agent.tools.main_crawl_data import crawl_data_tool
from AI_Agent.tools.save_data_to_csv_daily import save_data_tool
from AI_Agent.logs.checking_logs import log_workflow_step_tool
from AI_Agent.tools.time_series_analysis import time_series_analysis_core


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
workflow.add_node("log_after_crawl", lambda state: log_workflow_step_tool.invoke("Đã crawl dữ liệu"))
workflow.add_node("log_after_check", lambda state: log_workflow_step_tool.invoke("Đã kiểm tra preprocessing"))
workflow.add_node("log_after_fill", lambda state: log_workflow_step_tool.invoke("Đã fill dữ liệu bị thiếu"))
workflow.add_node("log_after_eda", lambda state: log_workflow_step_tool.invoke("Đã eda dữ liệu"))
workflow.add_node("log_after_save", lambda state: log_workflow_step_tool.invoke("Đã lưu dữ liệu xong"))
workflow.add_node("log_final", lambda state: log_workflow_step_tool.invoke("Workflow đã hoàn tất"))
workflow.add_node("log_after_baseline", lambda state: log_workflow_step_tool.invoke("Đã tính baseline"))

workflow.add_node("crawl_data", crawl_data_tool)
workflow.add_node("fill_missing", fill_missing_dates_tool)
workflow.add_node("test_node", test_output)
workflow.add_node("check_preprocessing_tool", check_preprocessing_tool)
workflow.add_node("save_data_node", save_data_tool)
workflow.add_node("eda_analysis", eda_tool)
workflow.add_node("time_series_analysis_core", time_series_analysis_core)
workflow.add_node("baseline_calculation", calculate_baseline_tool)

# 5. Kết nối các bước -------------------------------------------------------------------------------------------
workflow.set_entry_point("crawl_data")

workflow.add_edge("crawl_data", "log_after_crawl")
workflow.add_edge("log_after_crawl", "fill_missing")
workflow.add_edge("fill_missing", "log_after_fill")
workflow.add_edge("log_after_fill", "test_node")
workflow.add_edge("test_node", "check_preprocessing_tool")
workflow.add_edge("check_preprocessing_tool", "log_after_check")

# Nhánh song song từ log_after_check
workflow.add_edge("log_after_check", "save_data_node")
workflow.add_edge("log_after_check", "eda_analysis")

# Sau khi save và eda, gộp lại
workflow.add_edge("save_data_node", "log_after_save")
workflow.add_edge("eda_analysis", "log_after_eda")

# Gộp cả hai nhánh vào bước phân tích chuỗi thời gian
workflow.add_edge("log_after_save", "time_series_analysis_core")
workflow.add_edge("log_after_eda", "time_series_analysis_core")

# Sau time_series_analysis_core → tính baseline
workflow.add_edge("time_series_analysis_core", "baseline_calculation")
workflow.add_edge("baseline_calculation", "log_after_baseline")
workflow.add_edge("log_after_baseline", "log_final")  # ← Gộp luôn vào final log

# Kết thúc tại log_final
workflow.set_finish_point("log_final")

# 6. Compile graph ----------------------------------------------------------------------------------------------
graph = workflow.compile()

# 7. Chạy graph -------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    result = graph.invoke({})