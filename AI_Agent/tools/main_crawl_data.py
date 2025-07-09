from langchain_core.tools import tool
from AI_Agent.tools.crawl_data_agent import crawl_data

@tool
def crawl_data_tool(_input: dict = None):
    """
    Tự động thu thập dữ liệu của các mã cổ phiếu từ yfinance, Alpha Vantage hoặc file csv fallback.
    """
    df = crawl_data()
    return {"stock_data": df}