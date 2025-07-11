from langchain_core.tools import tool
import pandas as pd
import os
import random
from pydantic import BaseModel
from typing import Dict

class SplitInfo(BaseModel):
    splits: int
    output_path: str

class TuningInput(BaseModel):
    model_selection_result: Dict[str, str]  # ví dụ: {"AAPL": "GRU", "GOOG": "Prophet"}
    walk_forward_splits: Dict[str, SplitInfo]

@tool
def tuning_model_tool(input: TuningInput) -> dict:
    """
    Tự động tuning siêu tham số cho model đã chọn (GRU, Prophet, DeepLinear) dựa vào kết quả chọn model.
    """
    model_selection = input.model_selection_result
    split_info = input.walk_forward_splits

    output_dir = "/Users/minhtan/Documents/GitHub/Time_Series_Forecast/AI_Agent/output/tuning_output"
    os.makedirs(output_dir, exist_ok=True)

    all_results = {}

    for ticker, model_name in model_selection.items():
        file_path = split_info[ticker].output_path
        if not file_path or not os.path.exists(file_path):
            all_results[ticker] = {"error": "Không tìm thấy file split"}
            continue

        df = pd.read_csv(file_path, parse_dates=["Date"])
        df = df[df["type"] == "train"].sort_values("Date")

        if len(df) < 50:
            all_results[ticker] = {"error": "Dữ liệu quá ít"}
            continue

        # Tạo các tổ hợp hyperparameter giả lập (minh họa)
        results = []
        for i in range(n_trials):
            if model_name == "GRU":
                params = {
                    "hidden_size": random.choice([32, 64, 128]),
                    "num_layers": random.choice([1, 2]),
                    "dropout": round(random.uniform(0.1, 0.5), 2),
                    "lr": round(random.uniform(0.0005, 0.01), 4)
                }
            elif model_name == "Prophet":
                params = {
                    "seasonality_mode": random.choice(["additive", "multiplicative"]),
                    "changepoint_prior_scale": round(random.uniform(0.01, 0.5), 2)
                }
            else:  # DeepLinear
                params = {
                    "lr": round(random.uniform(0.0001, 0.005), 4),
                    "epochs": random.choice([50, 100, 150])
                }

            # Giả lập kết quả đánh giá (RMSE)
            score = round(random.uniform(1.0, 3.0), 3)
            results.append({"trial": i+1, "params": params, "RMSE": score})

        # Chuyển thành DataFrame
        rows = []
        for r in results:
            row = {"trial": r["trial"], "RMSE": r["RMSE"]}
            row.update(r["params"])
            rows.append(row)

        tuning_df = pd.DataFrame(rows)
        output_file = os.path.join(output_dir, f"{ticker}_{model_name}_tuning.csv")
        tuning_df.to_csv(output_file, index=False)

        all_results[ticker] = {
            "model": model_name,
            "best_score": tuning_df["RMSE"].min(),
            "best_params": tuning_df.loc[tuning_df["RMSE"].idxmin()].to_dict(),
            "tuning_file": output_file
        }

    return {"status": "done", "tuning_summary": all_results}
