import os
import pandas as pd
from AI_Agent.tools.models.PROPHET import prophet_model_tool
from AI_Agent.tools.models.GRU import gru_model_tool

def run_prophet_or_skip(state):
    output_dir = "/Users/minhtan/Documents/GitHub/Time_Series_Forecast/AI_Agent/output/output_models/prophet_results"
    walk_splits = state["walk_forward_splits"]

    summary = {}
    for ticker, info in walk_splits.items():
        result_path = os.path.join(output_dir, f"{ticker}_prophet_results.csv")
        if os.path.exists(result_path):
            print(f"[INFO] Đã có kết quả Prophet cho {ticker}, đọc lại từ file.")
            df_result = pd.read_csv(result_path)
            avg_metrics = df_result[["MAE", "RMSE", "MAPE"]].mean(numeric_only=True).to_dict()
            avg_metrics["ticker"] = ticker
            avg_metrics["result_path"] = result_path
            summary[ticker] = avg_metrics
        else:
            print(f"[INFO] Chưa có kết quả Prophet cho {ticker}, sẽ huấn luyện.")
            result = prophet_model_tool.invoke({
                "input": {
                    "walk_forward_splits": {ticker: info["output_path"]}
                }
            })
            summary[ticker] = result["summary"][ticker]

    return {**state, "prophet_summary": summary}

def run_gru_or_skip(state):
    output_dir = "/Users/minhtan/Documents/GitHub/Time_Series_Forecast/AI_Agent/output/output_models/gru_results"
    walk_splits = state["walk_forward_splits"]

    summary = {}
    for ticker, info in walk_splits.items():
        result_path = os.path.join(output_dir, f"{ticker}_gru_results.csv")
        if os.path.exists(result_path):
            print(f"[INFO] Đã có kết quả GRU cho {ticker}, đọc lại từ file.")
            df_result = pd.read_csv(result_path)
            avg_metrics = df_result[["MAE", "RMSE", "MAPE"]].mean(numeric_only=True).to_dict()
            avg_metrics["ticker"] = ticker
            avg_metrics["result_path"] = result_path
            summary[ticker] = avg_metrics
        else:
            print(f"[INFO] Chưa có kết quả GRU cho {ticker}, sẽ huấn luyện.")
            result = gru_model_tool.invoke({
                "input": {
                    "walk_forward_splits": {ticker: info["output_path"]}
                }
            })
            summary[ticker] = result["summary"][ticker]

    return {**state, "gru_summary": summary}

