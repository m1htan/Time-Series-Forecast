from typing import Dict
import pandas as pd
import numpy as np
import os
from statsmodels.tsa.stattools import adfuller, acf
from statsmodels.tsa.seasonal import seasonal_decompose


def choose_model_tool(state: dict) -> dict:
    """
    Xem xét dữ liệu đầu vào, đánh giá:
    - Tính dừng (stationarity)
    - Tính mùa vụ (seasonality)
    - Tính tương quan chuỗi (autocorrelation)

    Sau đó lựa chọn mô hình tốt nhất (Prophet, GRU, DeepLinear)
    và sử dụng kết quả dự đoán tương ứng từ model đã chạy trước đó.
    """

    walk_forward_splits = state.get("walk_forward_splits", {})
    if not walk_forward_splits:
        raise ValueError("Thiếu walk_forward_splits trong state")

    model_choices = {}
    chosen_model_per_ticker = {}
    prediction_results = {}
    evaluation_results = {}

    for ticker, info in walk_forward_splits.items():
        file_path = info["output_path"]

        if not os.path.exists(file_path):
            model_choices[ticker] = {"error": "Không tìm thấy file split"}
            continue

        df = pd.read_csv(file_path, parse_dates=["Date"])
        df = df[df["type"] == "train"].sort_values("Date")

        if len(df) < 50:
            model_choices[ticker] = {"error": "Không đủ dữ liệu train"}
            continue

        ts = df["Close"].dropna()

        # 1. Autocorrelation
        try:
            autocorr = acf(ts, nlags=10)[1]
        except Exception:
            autocorr = 0

        # 2. Seasonality
        try:
            decomposition = seasonal_decompose(ts, model="additive", period=7)
            seasonality_strength = np.var(decomposition.seasonal) / np.var(ts)
        except Exception:
            seasonality_strength = 0

        # 3. Stationarity (ADF)
        try:
            adf_pvalue = adfuller(ts)[1]
        except Exception:
            adf_pvalue = 1

        # 4. Chọn model
        if seasonality_strength > 0.15:
            chosen_model = "Prophet"
        elif adf_pvalue > 0.05 and autocorr > 0.5:
            chosen_model = "GRU"
        else:
            chosen_model = "DeepLinear"

        model_choices[ticker] = {
            "adf_pvalue": round(adf_pvalue, 4),
            "seasonality_strength": round(seasonality_strength, 3),
            "autocorr_lag1": round(autocorr, 3),
            "chosen_model": chosen_model
        }

        chosen_model_per_ticker[ticker] = chosen_model

        # 5. Lấy kết quả dự báo từ state
        pred_key = f"{chosen_model.lower()}_prediction"
        eval_key = f"{chosen_model.lower()}_eval"

        if pred_key in state and ticker in state[pred_key]:
            prediction_results[ticker] = state[pred_key][ticker]
        if eval_key in state and ticker in state[eval_key]:
            evaluation_results[ticker] = state[eval_key][ticker]

    # Ghi kết quả ra CSV
    output_df = pd.DataFrame(list(model_choices.values()))
    output_dir = "/Users/minhtan/Documents/GitHub/Time_Series_Forecast/AI_Agent/output/model_selection"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "model_selection_summary.csv")
    output_df.to_csv(output_path, index=False)

    # Cập nhật lại state
    state["chosen_model"] = chosen_model_per_ticker
    state["chosen_model_prediction"] = prediction_results
    state["chosen_model_eval"] = evaluation_results
    state["model_selection_detail"] = model_choices

    return state
