from datetime import datetime

import yfinance as yf
import pandas as pd
import os
from alpha_vantage.timeseries import TimeSeries
from dotenv import load_dotenv

load_dotenv(dotenv_path='/Users/minhtan/Documents/GitHub/Time_Series_Forecast/AI_Agent/config/config.env')

# Truy cập các biến môi trường
ALPHAVANTAGE_API_KEY = os.getenv("ALPHAVANTAGE_API_KEY")

# Fallback file
CSV_FALLBACK_PATH = "fallback_data.csv"

def crawl_from_yfinance(symbols, start, end):
    df = yf.download(symbols, start=start, end=end, group_by='ticker', auto_adjust=True)
    return df

def crawl_from_alpha_vantage(symbol, api_key="ALPHAVANTAGE_API_KEY"):
    ts = TimeSeries(key=api_key, output_format='pandas')
    data, _ = ts.get_daily(symbol=symbol, outputsize='compact')
    return data

def load_from_csv(path=CSV_FALLBACK_PATH):
    return pd.read_csv(path)

def crawl_data():
    end_date = datetime.today().strftime("%Y-%m-%d")
    try:
        df = crawl_from_yfinance(["AAPL", "MSFT", "GOOG"], start="2020-01-01", end=end_date)
        print("Dữ liệu đã lấy từ yfinance.")
    except Exception as e1:
        print(f"Lỗi yfinance: {e1}")
        try:
            df_list = []
            for symbol in ["AAPL", "MSFT", "GOOG"]:
                data = crawl_from_alpha_vantage(symbol)
                df_list.append(data)
            df = pd.concat(df_list, keys=["AAPL", "MSFT", "GOOG"])
            print("Dữ liệu lấy từ Alpha Vantage.")
        except Exception as e2:
            print(f"Lỗi Alpha Vantage: {e2}")
            df = load_from_csv()
            print("Dữ liệu load từ file CSV fallback.")

    return df
