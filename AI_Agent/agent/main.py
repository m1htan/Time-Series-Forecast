# ai_agent_stock/agent.py
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import initialize_agent, AgentType
from .tools import crawl_yfinance, crawl_alphavantage, load_csv

def build_agent(verbose: bool = True):
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-thinking-exp-1219",
        temperature=0.0,
        max_tokens=1024,
    )
    tools = [crawl_yfinance, crawl_alphavantage, load_csv]

    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent_type=AgentType.OPENAI_FUNCTIONS,  # tool‑calling chuẩn
        verbose=verbose,
    )
    return agent

if __name__ == "__main__":
    agent = build_agent()
    prompt = (
        "Crawl dữ liệu 3 mã AAPL, MSFT, GOOG giai đoạn 2020‑nay "
        "bằng yfinance và lưu vào bộ nhớ."
    )
    print(agent.invoke(prompt))
