from langchain_core.tools import tool
from datetime import datetime

@tool
def log_workflow_step_tool(step_description: str, level=None, metadata=None, step=None) -> dict:
    """
    Ghi log quá trình thực hiện workflow vào file `workflow_log.txt`.
    Ví dụ: "Đã crawl dữ liệu", "Đã fill missing", "Đã kiểm tra preprocessing".
    """
    print(f"[{level}] {step}: {step_description}")
    if metadata:
        print(f"Metadata: {metadata}")
        
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] {step_description}\n"

    log_path = "/Users/minhtan/Documents/GitHub/Time_Series_Forecast/AI_Agent/logs/workflow_log.csv"

    try:
        with open(log_path, "a") as f:
            f.write(log_entry)
        print(f"Đã ghi log: {step_description}")
    except Exception as e:
        print(f"Lỗi khi ghi log: {e}")

    return {"log": step_description}