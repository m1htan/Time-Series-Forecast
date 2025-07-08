# test_tools.py
import pandas as pd
from memory import DataFrameStore

def test_yfinance_single():
    DataFrameStore.clear()
    crawl_yfinance.invoke({"tickers": ["AAPL"]})
    df = DataFrameStore.get("AAPL")
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert {"Date", "Close"}.issubset(df.columns)

# test_agent.py
from agent import build_agent
from memory import DataFrameStore

def test_agent_workflow(monkeypatch):
    DataFrameStore.clear()
    agent = build_agent(verbose=False)
    agent.invoke("Hãy crawl AAPL, MSFT, GOOG bằng yfinance.")
    for tk in ["AAPL", "MSFT", "GOOG"]:
        assert DataFrameStore.get(tk) is not None
