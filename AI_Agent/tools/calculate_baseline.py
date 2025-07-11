from langchain_core.tools import tool
import pandas as pd
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

BASELINE_OUTPUT_DIR = "/Users/minhtan/Documents/GitHub/Time_Series_Forecast/AI_Agent/output/baseline_output"
os.makedirs(BASELINE_OUTPUT_DIR, exist_ok=True)

@tool
def calculate_baseline_tool(stock_data: object) -> dict:
    """
    Dự báo baseline bằng Persistence Model: Close_t = Close_(t-1)
    Lưu kết quả và tính MAE, RMSE cho từng mã.
    """
    result = {}

    for ticker, df in stock_data.items():
        df = df.copy()
        df = df.reset_index().rename(columns={"record_date": "Date"})
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.set_index("Date").sort_index()

        if "Close" not in df.columns or len(df) < 2:
            result[ticker] = {"error": "Không đủ dữ liệu hoặc thiếu cột Close"}
            continue

        df["prediction"] = df["Close"].shift(1)  # Persistence model
        df = df.dropna()

        y_true = df["Close"]
        y_pred = df["prediction"]

        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))

        # Lưu kết quả dự báo vào CSV
        df[["Close", "prediction"]].to_csv(os.path.join(BASELINE_OUTPUT_DIR, f"{ticker}_baseline.csv"))

        result[ticker] = {
            "MAE": mae,
            "RMSE": rmse,
            "n_samples": len(df)
        }

    return {"baseline_evaluation": result}
