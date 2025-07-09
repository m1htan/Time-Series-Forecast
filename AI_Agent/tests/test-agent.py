# test_tools.py
import pandas as pd
from AI_Agent.agent.tools import crawl_yfinance
from AI_Agent.memories.memory import DataFrameStore
from AI_Agent.agent.main import build_agent

def test_yfinance_single():
    DataFrameStore.clear()
    crawl_yfinance.invoke({"tickers": ["AAPL"]})
    df = DataFrameStore.get("AAPL")
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert {"Date", "Close"}.issubset(df.columns)

# test_agent.py
def test_agent_workflow(monkeypatch):
    DataFrameStore.clear()
    agent = build_agent(verbose=False)
    agent.invoke("Hãy crawl AAPL, MSFT, GOOG bằng yfinance.")
    for tk in ["AAPL", "MSFT", "GOOG"]:
        assert DataFrameStore.get(tk) is not None
