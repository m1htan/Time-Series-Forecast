import pandas as pd
from langchain_core.tools import tool

@tool
def fill_missing_dates_tool(stock_data: object) -> dict:
    """
    Tự động chèn các ngày bị thiếu vào chuỗi thời gian (theo lịch), và nội suy giá trị bị thiếu.
    Áp dụng cho từng mã cổ phiếu trong stock_data.
    """
    if stock_data is None:
        raise ValueError("Không có dữ liệu stock_data.")

    if callable(stock_data):
        stock_data = stock_data()

    df = stock_data
    result = {}

    if isinstance(df.columns, pd.MultiIndex):
        for ticker in df.columns.levels[0]:
            df_ticker = df[ticker].copy()
            df_ticker = df_ticker.reset_index().rename(columns={"Date": "record_date"})

            # Đặt lại index
            df_ticker = df_ticker.set_index('record_date').sort_index()

            # Tạo khoảng thời gian liên tục theo ngày
            full_range = pd.date_range(start=df_ticker.index.min(), end=df_ticker.index.max(), freq='D')
            df_filled = df_ticker.reindex(full_range)
            df_filled.index.name = 'record_date'

            # Xử lý nội suy cho các cột giá
            price_cols = ['Open', 'High', 'Low', 'Close']
            for col in price_cols:
                if col in df_filled.columns:
                    df_filled[col] = (
                        df_filled[col]
                        .interpolate(method='linear')
                        .bfill()
                        .ffill()
                    )

            # Xử lý Volume bằng forward fill
            if 'Volume' in df_filled.columns:
                df_filled['Volume'] = df_filled['Volume'].ffill()

            result[ticker] = df_filled
    else:
        raise ValueError("Dữ liệu đầu vào không đúng định dạng MultiIndex.")

    return {"stock_data": result}