import os
import warnings
import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

FORECAST_DAYS = 5
RESULTS_DIR = 'output/results_lgbm'
MODELS_DIR = 'output/models_lgbm'
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Load EDA output
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


def train_lightgbm_model(df, ticker_name, hyperparams):
    df_feat = generate_features(df)

    feature_cols = [col for col in df_feat.columns if col not in ['ds', 'y'] + [f'y_t+{i}' for i in range(1, FORECAST_DAYS + 1)]]
    target_cols = [f'y_t+{i}' for i in range(1, FORECAST_DAYS + 1)]

    X = df_feat[feature_cols]
    y = df_feat[target_cols]
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.1)

    model = lgb.LGBMRegressor(**hyperparams)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Save model
    model.booster_.save_model(f"{MODELS_DIR}/{ticker_name}_lgbm.txt")

    # Metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

    # Forecast future
    last_row = df_feat.iloc[-1:]
    last_input = last_row[feature_cols]
    future_preds = model.predict(last_input).flatten()
    future_dates = pd.date_range(start=last_row['ds'].values[0] + np.timedelta64(1, 'D'), periods=FORECAST_DAYS, freq='B')
    forecast = [{"ds": str(date.date()), "yhat": round(pred, 2)} for date, pred in zip(future_dates, future_preds)]

    return {
        'RMSE': round(rmse, 2),
        'MAE': round(mae, 2),
        'MAPE(%)': round(mape, 2),
        'Forecast': forecast
    }


def run_lightgbm_pipeline():
    params_dict = {
        'AAPL': {'num_leaves': 31, 'max_depth': 7, 'learning_rate': 0.05, 'n_estimators': 500, 'subsample': 0.8, 'colsample_bytree': 0.9},
        'MSFT': {'num_leaves': 50, 'max_depth': 10, 'learning_rate': 0.03, 'n_estimators': 700, 'subsample': 0.8, 'colsample_bytree': 0.9},
        'GOOG': {'num_leaves': 40, 'max_depth': 8, 'learning_rate': 0.04, 'n_estimators': 600, 'subsample': 0.8, 'colsample_bytree': 0.9},
    }

    print("\n====== LightGBM Forecasting Results ======")
    for ticker, params in params_dict.items():
        if ticker in df_all.columns:
            df = df_all[[ticker]].rename(columns={ticker: 'y'})
            df['ds'] = df.index
            results = train_lightgbm_model(df, ticker, params)

            print(f"\n[{ticker}]")
            print("MAE:", results['MAE'])
            print("RMSE:", results['RMSE'])
            print("MAPE(%):", results['MAPE(%)'])
            print("Next 5-Day Forecast:")
            for row in results['Forecast']:
                print(f"{row['ds']}: {row['yhat']}")


if __name__ == "__main__":
    run_lightgbm_pipeline()