import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split

# Set random seeds for reproducibility
import random
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

# Configs
FORECAST_DAYS = 5
INPUT_SEQ_LEN = 60
RESULTS_DIR = 'output/results_gru'
MODELS_DIR = 'output/models_gru'
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Load preprocessed data from EDA
df_all = pd.read_csv("output/stock_prices.csv", parse_dates=True, index_col=0)

forecast_results = {}

def prepare_data(series, input_len=60, target_len=5):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(series.values.reshape(-1, 1))

    X, y = [], []
    for i in range(input_len, len(scaled) - target_len):
        X.append(scaled[i - input_len:i, 0])
        y.append(scaled[i:i + target_len, 0])

    X, y = np.array(X), np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    return X, y, scaler

def build_gru_model(input_shape, units_1=64, units_2=32, dropout=0.2, lr=0.001):
    model = Sequential()
    model.add(GRU(units_1, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(dropout))
    model.add(GRU(units_2))
    model.add(Dropout(dropout))
    model.add(Dense(FORECAST_DAYS))
    model.compile(optimizer=Adam(learning_rate=lr), loss='mse')
    return model

def evaluate_and_plot(ticker, y_true, y_pred, history):
    # Metrics
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    print(f"[{ticker}] RMSE: {rmse:.2f}, MAE: {mae:.2f}, MAPE: {mape:.2f}%")

    # Plot prediction vs actual
    plt.figure(figsize=(10, 4))
    plt.plot(y_true[:100], label='Actual')
    plt.plot(y_pred[:100], label='Predicted')
    plt.title(f'{ticker} Forecast vs Actual')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/{ticker}_forecast.png")
    plt.close()

    # Plot loss
    plt.figure()
    plt.plot(history.history['loss'], label='Loss')
    plt.title(f'{ticker} Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/{ticker}_loss.png")
    plt.close()

    return rmse, mae, mape

def train_gru_for_ticker(ticker, config):
    print(f"\n=== Training GRU for {ticker} ===")
    series = df_all[ticker].dropna()
    X, y, scaler = prepare_data(series, input_len=INPUT_SEQ_LEN, target_len=FORECAST_DAYS)

    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.1)

    model = build_gru_model(X_train.shape[1:], **config)
    history = model.fit(X_train, y_train,
                        validation_data=(X_test, y_test),
                        epochs=config['epochs'],
                        batch_size=config['batch_size'],
                        verbose=0)

    preds = model.predict(X_test)
    preds_rescaled = scaler.inverse_transform(preds.reshape(-1, 1)).reshape(preds.shape)
    y_test_rescaled = scaler.inverse_transform(y_test.reshape(-1, 1)).reshape(y_test.shape)

    rmse, mae, mape = evaluate_and_plot(ticker, y_test_rescaled.flatten(), preds_rescaled.flatten(), history)

    model.save(f"{MODELS_DIR}/{ticker}_gru_model.h5")

    # Forecast next 5 days
    last_sequence = series.values[-INPUT_SEQ_LEN:]
    scaled_input = scaler.transform(last_sequence.reshape(-1, 1)).reshape(1, INPUT_SEQ_LEN, 1)
    forecast_scaled = model.predict(scaled_input)[0]
    forecast = scaler.inverse_transform(forecast_scaled.reshape(-1, 1)).flatten()

    forecast_dates = pd.date_range(start=series.index[-1] + pd.Timedelta(days=1), periods=FORECAST_DAYS, freq='B')
    forecast_output = [{"ds": str(date.date()), "yhat": round(value, 2)} for date, value in zip(forecast_dates, forecast)]

    forecast_results[ticker] = {
        'RMSE': round(rmse, 2),
        'MAE': round(mae, 2),
        'MAPE(%)': round(mape, 2),
        'Forecast': forecast_output
    }

# Configs per stock
model_configs = {
    'AAPL': {'units_1': 64, 'units_2': 32, 'dropout': 0.2, 'lr': 0.001, 'batch_size': 32, 'epochs': 50},
    'MSFT': {'units_1': 128, 'units_2': 64, 'dropout': 0.3, 'lr': 0.001, 'batch_size': 32, 'epochs': 60},
    'GOOG': {'units_1': 64, 'units_2': 32, 'dropout': 0.25, 'lr': 0.0008, 'batch_size': 32, 'epochs': 50}
}

if __name__ == '__main__':
    for ticker, cfg in model_configs.items():
        if ticker in df_all.columns:
            train_gru_for_ticker(ticker, cfg)

    print("\n====== GRU Forecasting Results ======")
    for ticker, res in forecast_results.items():
        print(f"\n[{ticker}]")
        print("MAE:", res['MAE'])
        print("RMSE:", res['RMSE'])
        print("MAPE(%):", res['MAPE(%)'])
        print("Next 5-Day Forecast:")
        for row in res['Forecast']:
            print(f"{row['ds']}: {row['yhat']}")
