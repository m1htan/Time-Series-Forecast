import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os
import warnings

warnings.filterwarnings("ignore")

# Load the unified stock data from EDA
DATA_PATH = "output/stock_prices.csv"
df_all = pd.read_csv(DATA_PATH, parse_dates=True, index_col=0)

FORECAST_DAYS = 5
TRAIN_RATIO = 0.8

results = {}

print("\n====== Persistence Forecasting Results ======")

for ticker in ['AAPL', 'MSFT', 'GOOG']:
    if ticker not in df_all.columns:
        continue

    df = df_all[[ticker]].dropna().copy()
    df.columns = ['y']
    df['ds'] = df.index
    df.reset_index(drop=True, inplace=True)

    n = len(df)
    train_size = int(n * TRAIN_RATIO)
    test_size = FORECAST_DAYS

    y_true_all = []
    y_pred_all = []

    for start in range(train_size, n - FORECAST_DAYS + 1):
        # Take last value in train as prediction base
        last_value = df.loc[start - 1, 'y']
        y_pred = [last_value] * FORECAST_DAYS
        y_true = df.loc[start:start + FORECAST_DAYS - 1, 'y'].tolist()

        y_true_all.extend(y_true)
        y_pred_all.extend(y_pred)

    y_true_all = np.array(y_true_all)
    y_pred_all = np.array(y_pred_all)

    rmse = np.sqrt(mean_squared_error(y_true_all, y_pred_all))
    mae = mean_absolute_error(y_true_all, y_pred_all)
    mape = np.mean(np.abs((y_true_all - y_pred_all) / y_true_all)) * 100

    results[ticker] = {
        'MAE': round(mae, 2),
        'RMSE': round(rmse, 2),
        'MAPE(%)': round(mape, 2),
        'LastForecast': y_pred_all[-FORECAST_DAYS:].tolist(),
        'LastDates': df['ds'].iloc[-FORECAST_DAYS:].dt.date.tolist()
    }

    print(f"\n[{ticker}]")
    print("MAE:", results[ticker]['MAE'])
    print("RMSE:", results[ticker]['RMSE'])
    print("MAPE(%):", results[ticker]['MAPE(%)'])
    print("Next 5-Day Forecast:")
    for date, val in zip(results[ticker]['LastDates'], results[ticker]['LastForecast']):
        print(f"{date}: {round(val, 2)}")