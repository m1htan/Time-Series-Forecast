from pydantic import BaseModel
from typing import Dict
from langchain_core.tools import tool
import numpy as np
import pandas as pd
import os

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import Input

class GRUModelInput(BaseModel):
    walk_forward_splits: Dict[str, str]
    model_config = {
        "arbitrary_types_allowed": True
    }

def mean_absolute_percentage_error(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    non_zero_idx = y_true != 0
    return np.mean(np.abs((y_true[non_zero_idx] - y_pred[non_zero_idx]) / y_true[non_zero_idx])) * 100

def build_gru_model(input_shape, units=64, dropout=0.3, lr=0.0005):
    """
    Xây dựng mô hình GRU với cấu trúc phù hợp cho dữ liệu tài chính.

    - units = 64: số lượng đơn vị ẩn cho lớp GRU đầu tiên.
    - dropout = 0.3: giảm overfitting, phù hợp với dữ liệu có nhiễu.
    - lr = 0.0005: learning rate nhỏ hơn để ổn định hơn trong học dữ liệu thời gian.

    Args:
        input_shape: tuple (window, n_features)
    """
    model = Sequential([
        Input(shape=input_shape),
        GRU(units, return_sequences=True),
        Dropout(dropout),
        GRU(units // 2),
        Dropout(dropout),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=lr), loss='mse')
    return model

def prepare_gru_data(df, feature_cols, target_col="Close", window=5, horizon=1):
    if len(df) < window + horizon:
        return np.array([]), np.array([]), None, None  # Trả về rỗng nếu không đủ dòng
    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()

    df = df.dropna().copy()

    if len(df) <= window + horizon:
        return None, None, None, None

    features_scaled = feature_scaler.fit_transform(df[feature_cols])
    target_scaled = target_scaler.fit_transform(df[[target_col]])

    X, y = [], []
    for i in range(window, len(df) - horizon):
        X.append(features_scaled[i-window:i])
        y.append(target_scaled[i + horizon - 1])

    return np.array(X), np.array(y), feature_scaler, target_scaler

@tool
def gru_model_tool(input: GRUModelInput) -> dict:
    """
    Huấn luyện GRU trên từng split được chia sẵn (walk-forward), đánh giá và lưu kết quả.
    """
    split_paths = input.walk_forward_splits
    if not split_paths:
        raise ValueError("[ERROR] Không có dữ liệu đầu vào cho GRU model (walk_forward_splits rỗng)")

    output_dir = "/Users/minhtan/Documents/GitHub/Time_Series_Forecast/AI_Agent/output/output_models/gru_results"
    os.makedirs(output_dir, exist_ok=True)

    summary = {}

    for ticker, file_path in split_paths.items():
        print(f"\n[INFO] Đọc dữ liệu từ: {file_path}")

        if not os.path.exists(file_path):
            print(f"[ERROR] File không tồn tại: {file_path}")
            continue

        try:
            df = pd.read_csv(file_path, parse_dates=["Date"])
            print(f"[CHECK] Cột trong file: {list(df.columns)}")
            print("[CHECK] 5 dòng đầu tiên:")
            print(df.head())
        except Exception as e:
            print(f"[ERROR] Lỗi khi đọc file CSV {file_path}: {e}")
            continue

        if df is None or df.empty:
            print(f"[WARN] Không có dữ liệu cho {ticker}, bỏ qua.")
            continue

        try:
            df = df[[
                'Date', 'Close', 'residual', 'lag_1', 'lag_3', 'rolling_mean_7',
                'rolling_std_14', 'log_diff', 'day_of_week', 'month', 'rsi_14',
                'macd', 'macd_signal', 'split_id', 'type'
            ]].dropna().sort_values("Date")
        except Exception as e:
            print(f"[ERROR] Không đủ cột cho ticker {ticker}: {e}")
            continue

        all_metrics = []

        for split_id in df["split_id"].unique():
            split_df = df[df["split_id"] == split_id].copy()
            train = split_df[split_df["type"] == "train"]
            test = split_df[split_df["type"] == "test"]

            print(f"[CHECK] Split {split_id} - train.shape={train.shape}, test.shape={test.shape}")

            if len(train) == 0 or len(test) == 0:
                print(f"[WARN] Split {split_id} không có đủ dữ liệu train/test, bỏ qua.")
                continue

            try:
                feature_cols = [
                    'Close', 'lag_1', 'lag_3', 'rolling_mean_7',
                    'rolling_std_14', 'log_diff', 'day_of_week',
                    'month', 'rsi_14', 'macd', 'macd_signal'
                ]

                X_train, y_train, feature_scaler, target_scaler = prepare_gru_data(train, feature_cols, window=5, horizon=1)

                if X_train is None or len(X_train) == 0:
                    raise ValueError(f"Dữ liệu train split {split_id} quá nhỏ hoặc không đủ")

                print(f"[DEBUG] X_train shape for {ticker} - split {split_id}: {X_train.shape}")

                X_test, y_test, _, _ = prepare_gru_data(test, feature_cols, window=5, horizon=1)

                if X_test is None or len(X_test) == 0:
                    raise ValueError(f"Dữ liệu test split {split_id} quá nhỏ hoặc không đủ")

                model = build_gru_model(input_shape=(X_train.shape[1], X_train.shape[2]))
                model.fit(
                    X_train, y_train,
                    epochs=50,
                    batch_size=8,
                    verbose=0,
                    callbacks=[
                        EarlyStopping(monitor='loss', patience=7, restore_best_weights=True)
                    ]
                )

                y_pred = model.predict(X_test)
                y_pred_inv = target_scaler.inverse_transform(y_pred)
                y_test_inv = target_scaler.inverse_transform(y_test)
                print(f"y_pred sample: {y_pred[:5].flatten()}, min: {y_pred.min()}, max: {y_pred.max()}")

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

            except Exception as e:
                all_metrics.append({
                    "ticker": ticker,
                    "split_id": split_id,
                    "error": str(e)
                })

        try:
            df_result = pd.DataFrame(all_metrics)
            result_path = os.path.join(output_dir, f"{ticker}_gru_results.csv")
            df_result.to_csv(result_path, index=False)
            print(f"[INFO] Đã lưu kết quả: {result_path}")
        except Exception as e:
            print(f"[ERROR] Lỗi khi lưu file {result_path}: {e}")
            continue

        avg_metrics = df_result[["MAE", "RMSE", "MAPE"]].mean(numeric_only=True).to_dict()
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