from langchain_core.tools import tool

@tool
def save_data_tool(stock_data: object) -> dict:
    """
    Lưu dữ liệu từ stock_data thành các file CSV riêng biệt theo mã cổ phiếu vào thư mục 'data/'.
    """
    import os
    from datetime import datetime

    df = stock_data
    if df is None:
        raise ValueError("Không tìm thấy dữ liệu 'stock_data'.")

    base_dir = "/Users/minhtan/Documents/GitHub/Time_Series_Forecast/AI_Agent/data_update_daily"
    os.makedirs(base_dir, exist_ok=True)
    today = datetime.today().strftime("%Y-%m-%d")

    if hasattr(df, "columns") and isinstance(df.columns, pd.MultiIndex):
        for ticker in df.columns.levels[0]:
            sub_df = df[ticker].copy()
            file_path = f"/Users/minhtan/Documents/GitHub/Time_Series_Forecast/AI_Agent/data_update_daily/{ticker}_{today}.csv"
            sub_df.to_csv(file_path)
            print(f"Đã lưu dữ liệu {ticker} vào {file_path}")
    else:
        for ticker in ["AAPL", "MSFT", "GOOG"]:
            if ticker in df:
                sub_df = df[ticker].copy()
                file_path = f"/Users/minhtan/Documents/GitHub/Time_Series_Forecast/AI_Agent/data_update_daily/{ticker}.csv"
                sub_df.to_csv(file_path)
                print(f"Đã lưu dữ liệu {ticker} vào {file_path}")
            else:
                print(f"Không tìm thấy dữ liệu cho {ticker}.")

    return {"stock_data": df}