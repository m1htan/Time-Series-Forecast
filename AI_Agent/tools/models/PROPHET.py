from pydantic import BaseModel
from typing import Dict
from langchain_core.tools import tool
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import os
import pandas as pd

class ProphetModelInput(BaseModel):
    walk_forward_splits: Dict[str, str]  # ticker -> path to CSV

    model_config = {
        "arbitrary_types_allowed": True
    }

@tool
def prophet_model_tool(input: ProphetModelInput) -> dict:
    """
    Huấn luyện Prophet trên từng split được chia sẵn (walk-forward), đánh giá và lưu kết quả.
    """
    split_paths = input.walk_forward_splits
    output_dir = "/Users/minhtan/Documents/GitHub/Time_Series_Forecast/AI_Agent/output/output_models/prophet_results"
    os.makedirs(output_dir, exist_ok=True)

    summary = {}

    for ticker, file_path in split_paths.items():
        print(f"\n[INFO] Đọc dữ liệu từ: {file_path}")

        if not os.path.exists(file_path):
            print(f"[ERROR] File không tồn tại: {file_path}")
            continue

        try:
            # Kiểm tra cột và in 5 dòng đầu
            preview_df = pd.read_csv(file_path, nrows=5)
            print(f"[CHECK] Cột trong file: {list(preview_df.columns)}")
            print("[CHECK] 5 dòng đầu tiên:")
            print(preview_df)

            # Bắt đầu đọc full dữ liệu
            df = pd.read_csv(file_path, parse_dates=["record_date"])
        except Exception as e:
            print(f"[ERROR] Lỗi khi đọc file CSV {file_path}: {e}")
            continue

        df = pd.read_csv(file_path, parse_dates=["record_date"])
        df.rename(columns={"record_date": "ds", "Close": "y"}, inplace=True)
        df = df[["ds", "y", "split_id", "type"]].dropna().sort_values("ds")

        all_metrics = []

        for split_id in df["split_id"].unique():
            split_df = df[df["split_id"] == split_id]
            train = split_df[split_df["type"] == "train"]
            test = split_df[split_df["type"] == "test"]

            if len(train) == 0 or len(test) == 0:
                continue

            try:
                model = Prophet(daily_seasonality=True)
                model.fit(train[["ds", "y"]])

                forecast = model.predict(test[["ds"]])
                y_true = test["y"].values
                y_pred = forecast["yhat"].values

                mae = mean_absolute_error(y_true, y_pred)
                rmse = np.sqrt(mean_squared_error(y_true, y_pred))
                mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

                all_metrics.append({
                    "ticker": ticker,
                    "split_id": split_id,
                    "MAE": mae,
                    "RMSE": rmse,
                    "MAPE": mape
                })

            except Exception as e:
                all_metrics.append({
                    "ticker": ticker,
                    "split_id": split_id,
                    "error": str(e)
                })

        df_result = pd.DataFrame(all_metrics)
        result_path = os.path.join(output_dir, f"{ticker}_prophet_results.csv")
        df_result.to_csv(result_path, index=False)

        avg_metrics = df_result[["MAE", "RMSE", "MAPE"]].mean().to_dict()
        avg_metrics["ticker"] = ticker
        avg_metrics["result_path"] = result_path
        summary[ticker] = avg_metrics

    summary_df = pd.DataFrame.from_dict(summary, orient="index")
    summary_path = os.path.join(output_dir, "summary_all_tickers.csv")
    summary_df.to_csv(summary_path)

    return {
        "status": "success",
        "summary": summary,
        "output_dir": output_dir
    }