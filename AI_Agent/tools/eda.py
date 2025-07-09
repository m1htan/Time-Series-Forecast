import os
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import seaborn as sns
from statsmodels.tsa.stattools import adfuller, acf, pacf
from datetime import datetime
from langchain_core.tools import tool

from AI_Agent.logs.checking_logs import log_workflow_step_tool

EDA_OUTPUT_DIR = "/Users/minhtan/Documents/GitHub/Time_Series_Forecast/AI_Agent/eda_output"
os.makedirs(EDA_OUTPUT_DIR, exist_ok=True)

@tool
def eda_tool(stock_data: object) -> dict:
    """
    Phân tích EDA trên mỗi mã cổ phiếu, lưu hình ảnh và thống kê mô tả vào thư mục riêng theo từng mã.
    """
    if stock_data is None:
        raise ValueError("Không có dữ liệu stock_data.")

    if callable(stock_data):
        stock_data = stock_data()

    result = {}

    for ticker, df in stock_data.items():
        df = df.copy()
        df = df.reset_index().rename(columns={"record_date": "Date"})
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.set_index("Date").sort_index()

        # Tạo thư mục riêng cho từng mã
        ticker_dir = os.path.join(EDA_OUTPUT_DIR, ticker)
        os.makedirs(ticker_dir, exist_ok=True)

        # 1. Lưu thống kê mô tả
        desc_path = os.path.join(ticker_dir, f"{ticker}_desc.csv")
        df.describe().to_csv(desc_path)

        try:
            # 2. Missing values heatmap
            plt.figure(figsize=(10, 2))
            sns.heatmap(df.isna(), cbar=False, cmap="viridis")
            plt.title(f"{ticker} - Missing Data")
            plt.tight_layout()
            plt.savefig(os.path.join(ticker_dir, f"{ticker}_missing.png"))
            plt.close()

            # 3. Boxplot
            plt.figure(figsize=(10, 4))
            sns.boxplot(data=df)
            plt.title(f"{ticker} - Boxplot")
            plt.tight_layout()
            plt.savefig(os.path.join(ticker_dir, f"{ticker}_boxplot.png"))
            plt.close()

            # 4. Histogram
            df.hist(figsize=(10, 6), bins=30)
            plt.suptitle(f"{ticker} - Histograms")
            plt.tight_layout()
            plt.savefig(os.path.join(ticker_dir, f"{ticker}_hist.png"))
            plt.close()

            # 5. Time series plot
            if "Close" in df.columns:
                plt.figure(figsize=(12, 5))
                df["Close"].plot()
                plt.title(f"{ticker} - Time Series (Close)")
                plt.savefig(os.path.join(ticker_dir, f"{ticker}_timeseries.png"))
                plt.close()

            # 6. ACF/PACF
            if "Close" in df.columns:
                from matplotlib import rcParams
                rcParams.update({'axes.grid': True})
                fig, axes = plt.subplots(1, 2, figsize=(14, 4))
                axes[0].stem(acf(df["Close"].dropna(), nlags=30), use_line_collection=True)
                axes[0].set_title("ACF")

                axes[1].stem(pacf(df["Close"].dropna(), nlags=30), use_line_collection=True)
                axes[1].set_title("PACF")

                plt.suptitle(f"{ticker} - ACF/PACF")
                plt.tight_layout()
                plt.savefig(os.path.join(ticker_dir, f"{ticker}_acf_pacf.png"))
                plt.close()

            # 7. ADF Test
            if "Close" in df.columns:
                adf_result = adfuller(df["Close"].dropna())
                result[ticker] = {
                    "adf_statistic": adf_result[0],
                    "p_value": adf_result[1],
                    "critical_values": adf_result[4]
                }

        except Exception as e:
            log_workflow_step_tool.invoke({
                "step": "eda_tool",
                "step_description": f"Lỗi khi xử lý EDA cho {ticker}",
                "level": "ERROR",
                "metadata": {"error": str(e)}
            })
        else:
            log_workflow_step_tool.invoke({
                "step": "eda_tool",
                "step_description": f"Đã xuất các biểu đồ và thống kê EDA cho {ticker}"
            })

    return {"eda_result": result}
