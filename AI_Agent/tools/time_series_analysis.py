from langchain_core.tools import tool
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import acf, pacf, adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
import ta

@tool
def time_series_analysis_core(stock_data: object) -> dict:
    """
    Phân tích định lượng chuỗi Close: autocorrelation, stationarity, decomposition, feature creation.
    """
    output = {}
    for ticker, df in stock_data.items():
        ticker_result = {}

        df = df.copy()
        df = df.reset_index().rename(columns={"record_date": "Date"})
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.set_index("Date").sort_index()
        df = df[["Close"]].dropna()

        # [2] ACF & PACF
        acf_vals = acf(df["Close"], nlags=30, fft=False)
        pacf_vals = pacf(df["Close"], nlags=30)

        ticker_result["acf_first_10"] = acf_vals[:10].tolist()
        ticker_result["pacf_first_10"] = pacf_vals[:10].tolist()
        ticker_result["acf_abs_sum"] = float(np.sum(np.abs(acf_vals[1:11])))

        # [3] White noise test
        ticker_result["is_white_noise"] = ticker_result["acf_abs_sum"] < 1

        # [4] Decomposition
        decomposition = seasonal_decompose(df["Close"], model="additive", period=30)
        trend = decomposition.trend
        seasonal = decomposition.seasonal
        residual = decomposition.resid

        df["residual"] = residual

        # [5] Detrended = residual
        df_clean = df.dropna()

        # [6] ADF Test
        adf_stat, pval, _, _, crit_vals, _ = adfuller(df_clean["residual"])
        ticker_result["adf_statistic"] = float(adf_stat)
        ticker_result["adf_pvalue"] = float(pval)
        ticker_result["adf_critical_values"] = crit_vals
        ticker_result["is_stationary"] = pval < 0.05

        # [7] Feature Engineering
        df_clean = df_clean.copy()
        df_clean.loc[:, "lag_1"] = df_clean["Close"].shift(1)
        df_clean.loc[:, "lag_3"] = df_clean["Close"].shift(3)
        df_clean.loc[:, "rolling_mean_7"] = df_clean["Close"].rolling(7).mean()
        df_clean.loc[:, "rolling_std_14"] = df_clean["Close"].rolling(14).std()
        df_clean.loc[:, "log_diff"] = np.log(df_clean["Close"]).diff()
        df_clean.loc[:, "day_of_week"] = df_clean.index.dayofweek
        df_clean.loc[:, "month"] = df_clean.index.month

        # RSI, MACD (nếu cần)
        try:
            df_clean = df_clean.copy()
            df_clean.loc[:, "rsi_14"] = ta.momentum.RSIIndicator(df_clean["Close"], window=14).rsi()
            macd = ta.trend.MACD(df_clean["Close"])
            df_clean.loc[:, "macd"] = macd.macd()
            df_clean.loc[:, "macd_signal"] = macd.macd_signal()
        except Exception as e:
            ticker_result["tech_indicator_error"] = str(e)

        df_clean.dropna().to_csv(f"/Users/minhtan/Documents/GitHub/Time_Series_Forecast/AI_Agent/output/timeseries_analysis_output/{ticker}_features.csv")
        output[ticker] = ticker_result

    return {
        "time_series_analysis_summary": output,
        "preprocessed_data": stock_data
    }