from pydantic import BaseModel
from typing import Dict
from langchain_core.tools import tool
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.callbacks import EarlyStopping

class DeepLinearModelInput(BaseModel):
    walk_forward_splits: Dict[str, str]
    model_config = {
        "arbitrary_types_allowed": True
    }

def mean_absolute_percentage_error(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    non_zero_idx = y_true != 0
    return np.mean(np.abs((y_true[non_zero_idx] - y_pred[non_zero_idx]) / y_true[non_zero_idx])) * 100

def prepare_deeplinear_data(df, feature_cols, target_col="Close", window=5, horizon=1):
    if len(df) < window + horizon:
        return np.array([]), np.array([]), None, None
    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()

    df = df.dropna().copy()
    if len(df) <= window + horizon:
        return None, None, None, None

    features_scaled = feature_scaler.fit_transform(df[feature_cols])
    target_scaled = target_scaler.fit_transform(df[[target_col]])

    X, y = [], []
    for i in range(window, len(df) - horizon):
        X.append(features_scaled[i - window:i].flatten())
        y.append(target_scaled[i + horizon - 1])

    return np.array(X), np.array(y), feature_scaler, target_scaler

@tool
def deeplinear_model_tool(input: DeepLinearModelInput) -> dict:
    """
    Huấn luyện mô hình DeepLinear (MLP) trên từng split (walk-forward), đánh giá và lưu kết quả.
    """
    split_paths = input.walk_forward_splits
    output_dir = "/Users/minhtan/Documents/GitHub/Time_Series_Forecast/AI_Agent/output/output_models/deeplinear_results"
    os.makedirs(output_dir, exist_ok=True)

    summary = {}
    for ticker, file_path in split_paths.items():
        if not os.path.exists(file_path):
            continue

        df = pd.read_csv(file_path, parse_dates=["Date"])
        df = df[[
            'Date', 'Close', 'residual', 'lag_1', 'lag_3', 'rolling_mean_7',
            'rolling_std_14', 'log_diff', 'day_of_week', 'month', 'rsi_14',
            'macd', 'macd_signal', 'split_id', 'type']].dropna().sort_values("Date")

        all_metrics = []
        for split_id in df["split_id"].unique():
            split_df = df[df["split_id"] == split_id]
            train = split_df[split_df["type"] == "train"]
            test = split_df[split_df["type"] == "test"]

            feature_cols = [
                'Close', 'lag_1', 'lag_3', 'rolling_mean_7', 'rolling_std_14',
                'log_diff', 'day_of_week', 'month', 'rsi_14', 'macd', 'macd_signal']

            X_train, y_train, feature_scaler, target_scaler = prepare_deeplinear_data(train, feature_cols)
            X_test, y_test, _, _ = prepare_deeplinear_data(test, feature_cols)

            if X_train is None or len(X_train) == 0 or X_test is None or len(X_test) == 0:
                continue

            model = Sequential([
                Input(shape=(X_train.shape[1],)),
                Dense(64, activation='relu'),
                Dense(32, activation='relu'),
                Dense(1)
            ])
            model.compile(optimizer='adam', loss='mse')
            model.fit(
                X_train, y_train,
                epochs=50,
                batch_size=8,
                verbose=0,
                callbacks=[EarlyStopping(monitor='loss', patience=7, restore_best_weights=True)]
            )

            y_pred = model.predict(X_test)
            y_pred_inv = target_scaler.inverse_transform(y_pred)
            y_test_inv = target_scaler.inverse_transform(y_test)

            mae = mean_absolute_error(y_test_inv, y_pred_inv)
            rmse = np.sqrt(mean_squared_error(y_test_inv, y_pred_inv))
            mape = mean_absolute_percentage_error(y_test_inv, y_pred_inv)

            all_metrics.append({
                "ticker": ticker,
                "split_id": split_id,
                "MAE": mae,
                "RMSE": rmse,
                "MAPE": mape
            })

        df_result = pd.DataFrame(all_metrics)
        result_path = os.path.join(output_dir, f"{ticker}_deeplinear_results.csv")
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