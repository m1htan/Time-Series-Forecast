# ai_agent_stock/tools.py
from typing import List
import os
import pandas as pd
import yfinance as yf
from alpha_vantage.timeseries import TimeSeries
from langchain_core.tools import tool
from .memory import DataFrameStore

# ---------- 1. Yahoo Finance ----------
@tool
def crawl_yfinance(tickers: List[str], period: str = "5y", interval: str = "1d") -> str:
    """
    Lấy OHLCV từ Yahoo Finance và lưu vào DataFrameStore.
    """
    for tk in tickers:
        df = yf.download(tk, period=period, interval=interval, auto_adjust=True)
        df.reset_index(inplace=True)
        DataFrameStore.save(tk, df)
    return f"Đã crawl {len(tickers)} mã bằng yfinance."

# ---------- 2. Alpha Vantage ----------
@tool
def crawl_alphavantage(ticker: str, output_size: str = "full") -> str:
    """
    Lấy dữ liệu daily Adjusted từ Alpha Vantage (dùng khi Yahoo bị rate‑limit).
    """
    ts = TimeSeries(key=os.getenv("ALPHAVANTAGE_API_KEY"), output_format="pandas")
    data, _ = ts.get_daily_adjusted(ticker, outputsize=output_size)
    data.sort_index(inplace=True)
    data.reset_index(inplace=True)
    DataFrameStore.save(ticker, data)
    return f"Đã crawl {ticker} bằng Alpha Vantage."

# ---------- 3. Fallback CSV ----------
@tool
def load_csv(path: str, ticker: str) -> str:
    """
    Đọc file CSV cục bộ (fallback) và lưu vào DataFrameStore.
    """
    df = pd.read_csv(path, parse_dates=["Date"])
    DataFrameStore.save(ticker, df)
    return f"Đã nạp dữ liệu {ticker} từ file CSV."
