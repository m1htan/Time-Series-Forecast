import os
import warnings
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

warnings.filterwarnings("ignore")
FORECAST_DAYS = 5
RESULTS_DIR = 'output/results_deeplinear'
MODELS_DIR = 'output/models_deeplinear'
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Load unified stock data (output of EDA)
df_all = pd.read_csv("output/stock_prices.csv", parse_dates=True, index_col=0)


def generate_features(df, window_config=None):
    df = df.copy()
    df.set_index('ds', inplace=True)

    if window_config is None:
        window_config = {
            'lag': [1, 3, 5, 7],
            'rolling_mean': [5, 10],
            'rolling_std': [5],
            'momentum': [3],
            'ema': [10]
        }

    for l in window_config['lag']:
        df[f'lag_{l}'] = df['y'].shift(l)
    for w in window_config['rolling_mean']:
        df[f'rolling_mean_{w}'] = df['y'].rolling(window=w).mean()
    for w in window_config['rolling_std']:
        df[f'rolling_std_{w}'] = df['y'].rolling(window=w).std()

    delta = df['y'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    df['RSI_14'] = 100 - (100 / (1 + rs))

    ema12 = df['y'].ewm(span=12, adjust=False).mean()
    ema26 = df['y'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_hist'] = df['MACD'] - df['MACD_signal']

    sma20 = df['y'].rolling(window=20).mean()
    std20 = df['y'].rolling(window=20).std()
    df['bb_upper'] = sma20 + 2 * std20
    df['bb_lower'] = sma20 - 2 * std20

    for w in window_config['ema']:
        df[f'EMA_{w}'] = df['y'].ewm(span=w, adjust=False).mean()
    for m in window_config['momentum']:
        df[f'momentum_{m}'] = df['y'] - df['y'].shift(m)

    for step in range(1, FORECAST_DAYS + 1):
        df[f'y_t+{step}'] = df['y'].shift(-step)

    df.dropna(inplace=True)
    df.reset_index(inplace=True)
    return df


def build_deep_model(input_dim, ticker):
    model = Sequential()
    if ticker == 'AAPL':
        model.add(Dense(128, activation='relu', input_shape=(input_dim,)))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))
        model.add(Dense(64, activation='relu'))
    elif ticker == 'MSFT':
        model.add(Dense(128, activation='relu', input_shape=(input_dim,)))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))
        model.add(Dense(64, activation='relu'))
    else:  # GOOG or others
        model.add(Dense(256, activation='relu', input_shape=(input_dim,)))
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.35))
        model.add(Dense(64, activation='relu'))

    model.add(Dense(FORECAST_DAYS))
    model.compile(optimizer='adam', loss='mse')
    return model


def train_and_forecast_deep_linear(df, ticker):
    df_feat = generate_features(df)
    X_cols = [col for col in df_feat.columns if col not in ['ds', 'y'] + [f'y_t+{i}' for i in range(1, FORECAST_DAYS + 1)]]
    y_cols = [f'y_t+{i}' for i in range(1, FORECAST_DAYS + 1)]

    X = df_feat[X_cols].values
    y = df_feat[y_cols].values

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, shuffle=False, test_size=0.1)

    model = build_deep_model(X_train.shape[1], ticker)
    model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0, validation_split=0.1)
    model.save(f"{MODELS_DIR}/{ticker}_deeplinear.h5")

    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

    # Forecast future
    X_last = X_scaled[-1].reshape(1, -1)
    y_future = model.predict(X_last).flatten()
    last_date = df_feat['ds'].values[-1]
    future_dates = pd.date_range(start=pd.to_datetime(last_date) + pd.Timedelta(days=1), periods=FORECAST_DAYS, freq='B')
    forecast = [{"ds": str(date.date()), "yhat": round(pred, 2)} for date, pred in zip(future_dates, y_future)]

    return {
        'MAE': round(mae, 2),
        'RMSE': round(rmse, 2),
        'MAPE(%)': round(mape, 2),
        'Forecast': forecast
    }


def run_deeplinear_pipeline():
    print("\n====== DeepLinear Forecasting Results ======")
    for ticker in ['AAPL', 'MSFT', 'GOOG']:
        if ticker in df_all.columns:
            df = df_all[[ticker]].rename(columns={ticker: 'y'})
            df['ds'] = df.index
            results = train_and_forecast_deep_linear(df, ticker)

            print(f"\n[{ticker}]")
            print("MAE:", results['MAE'])
            print("RMSE:", results['RMSE'])
            print("MAPE(%):", results['MAPE(%)'])
            print("Next 5-Day Forecast:")
            for row in results['Forecast']:
                print(f"{row['ds']}: {row['yhat']}")


if __name__ == "__main__":
    run_deeplinear_pipeline()