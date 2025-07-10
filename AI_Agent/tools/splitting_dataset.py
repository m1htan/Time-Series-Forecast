from langchain_core.tools import tool
import os
import pandas as pd
from AI_Agent.logs.checking_logs import log_workflow_step_tool

@tool
def walk_forward_split_tool(preprocessed_data: object, train_window: int = 200, forecast_horizon: int = 5) -> dict:
    """
    Áp dụng walk-forward validation trên mỗi mã cổ phiếu.
    Lưu toàn bộ train/test splits thành 1 file CSV duy nhất cho mỗi mã.
    """
    if callable(preprocessed_data):
        preprocessed_data = preprocessed_data()

    if not isinstance(preprocessed_data, dict):
        raise ValueError("Dữ liệu phải là dict[ticker: DataFrame]")

    result = {}
    output_dir = "/Users/minhtan/Documents/GitHub/Time_Series_Forecast/output/splits"
    os.makedirs(output_dir, exist_ok=True)

    for ticker, df in preprocessed_data.items():
        try:
            df = df.copy().dropna().sort_index()
            combined = []
            split_num = 1

            max_start = len(df) - train_window - forecast_horizon + 1
            for i in range(0, max_start, forecast_horizon):
                train = df.iloc[i:i+train_window].copy()
                test = df.iloc[i+train_window:i+train_window+forecast_horizon].copy()

                if len(train) < train_window or len(test) < forecast_horizon:
                    continue

                # Gắn nhãn
                train["split_id"] = split_num
                train["type"] = "train"

                test["split_id"] = split_num
                test["type"] = "test"

                combined.append(train)
                combined.append(test)
                split_num += 1

            # Gộp lại toàn bộ
            final_df = pd.concat(combined)
            final_df.to_csv(os.path.join(output_dir, f"{ticker}_walkforward.csv"))

            result[ticker] = {
                "splits": split_num - 1,
                "output_path": os.path.join(output_dir, f"{ticker}_walkforward.csv")
            }

            log_workflow_step_tool.invoke({
                "step": "walk_forward_split_tool",
                "step_description": f"Đã chia và lưu splits cho {ticker}",
                "metadata": {"splits": split_num - 1}
            })

        except Exception as e:
            log_workflow_step_tool.invoke({
                "step": "walk_forward_split_tool",
                "step_description": f"Lỗi khi chia splits cho {ticker}",
                "level": "ERROR",
                "metadata": {"error": str(e)}
            })
            raise e

    return {"walk_forward_splits": result}
