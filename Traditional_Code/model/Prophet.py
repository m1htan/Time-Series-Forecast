import pandas as pd
import numpy as np
import os
import json
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings("ignore")

# Load CSV
df = pd.read_csv("output/stock_prices.csv", index_col=0, parse_dates=True)

# Config
forecast_days = 5
param_grid = {
    'changepoint_prior_scale': [0.01, 0.05],
    'seasonality_mode': ['additive', 'multiplicative'],
    'holidays_prior_scale': [0.1]
}
output_dir = "output/prophet_results"
os.makedirs(output_dir, exist_ok=True)

def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def walk_forward_split(df, forecast_days):
    return df[:-forecast_days], df[-forecast_days:]

results = {}

for ticker in df.columns:
    print(f"\nBuilding Prophet model for {ticker}...")

    data = df[[ticker]].dropna().reset_index()
    data.columns = ['ds', 'y']
    train_df, test_df = walk_forward_split(data, forecast_days)

    best_rmse = float("inf")
    best_params = None
    best_model = None

    # Grid Search
    for cps in param_grid['changepoint_prior_scale']:
        for sm in param_grid['seasonality_mode']:
            for hps in param_grid['holidays_prior_scale']:
                model = Prophet(
                    changepoint_prior_scale=cps,
                    seasonality_mode=sm,
                    holidays_prior_scale=hps,
                    weekly_seasonality=True,
                    yearly_seasonality=(ticker == 'AAPL')
                )
                try:
                    model.fit(train_df)
                    future = model.make_future_dataframe(periods=forecast_days)
                    forecast = model.predict(future)

                    y_pred = forecast[['ds', 'yhat']].tail(forecast_days)['yhat'].values
                    y_true = test_df['y'].values
                    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

                    if rmse < best_rmse:
                        best_rmse = rmse
                        best_params = {'changepoint_prior_scale': cps, 'seasonality_mode': sm, 'holidays_prior_scale': hps}
                        best_model = model
                except Exception as e:
                    continue

    # Re-train with full data and best params
    final_model = Prophet(**best_params,
                          weekly_seasonality=True,
                          yearly_seasonality=(ticker == 'AAPL'))
    final_model.fit(data)
    future = final_model.make_future_dataframe(periods=forecast_days)
    forecast = final_model.predict(future)

    y_true = test_df['y'].values
    y_pred = forecast[['ds', 'yhat']].tail(forecast_days)['yhat'].values

    # Evaluate
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape_val = mape(y_true, y_pred)

    results[ticker] = {
        'BestParams': best_params,
        'MAE': round(mae, 2),
        'RMSE': round(rmse, 2),
        'MAPE(%)': round(mape_val, 2),
        'Forecast': forecast[['ds', 'yhat']].tail(forecast_days).to_dict(orient='records')
    }

    # Save forecast plot
    fig = final_model.plot(forecast)
    fig.savefig(f"{output_dir}/{ticker}_forecast.png")
    fig_comp = final_model.plot_components(forecast)
    fig_comp.savefig(f"{output_dir}/{ticker}_components.png")

    # Save forecast CSV
    forecast[['ds', 'yhat']].tail(forecast_days).to_csv(f"{output_dir}/{ticker}_forecast.csv", index=False)

# Save summary results
with open(f"{output_dir}/summary_results.json", "w") as f:
    json.dump(results, f, indent=4)

# Print
print("\n=== Prophet Forecast Summary ===")
for ticker, res in results.items():
    print(f"\n[{ticker}]")
    print("Best Params:", res['BestParams'])
    print("MAE:", res['MAE'])
    print("RMSE:", res['RMSE'])
    print("MAPE(%):", res['MAPE(%)'])
    print("Forecast (next 5 days):")
    for row in res['Forecast']:
        print(f"{row['ds'][:10]}: {round(row['yhat'], 2)}")
