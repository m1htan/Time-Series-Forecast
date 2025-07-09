import pandas as pd
from langchain_core.tools import tool

@tool
def check_preprocessing_tool(stock_data: object) -> dict:
    """
    Kiểm tra dữ liệu đầu vào trước xử lý:
    - Null / NaN
    - Thiếu dữ liệu thứ 7, CN
    - Dữ liệu trùng lặp
    - Sai định dạng ngày
    Ghi chú: chưa thực hiện xử lý, chỉ cảnh báo và trả log.
    """
    if stock_data is None:
        raise ValueError("Không có dữ liệu stock_data.")

    if callable(stock_data):
        stock_data = stock_data()

    df = stock_data
    logs = {}

    if isinstance(df, dict):  # Sau fill_missing_dates_tool, kiểu dữ liệu là dict[ticker] = DataFrame
        for ticker, df_ticker in df.items():
            df_check = df_ticker.copy()
            df_check = df_check.reset_index().rename(columns={"index": "record_date"})
            df_check['record_date'] = pd.to_datetime(df_check['record_date'], errors='coerce')

            log = {}

            # 1. Check null / NaN
            null_summary = df_check.isnull().sum().to_dict()
            log["missing_values"] = null_summary

            # 2. Check thiếu T7 & CN
            df_check['weekday'] = df_check['record_date'].dt.dayofweek  # 0: Monday, 6: Sunday
            missing_weekdays = set([5, 6]) - set(df_check['weekday'].unique())
            log["missing_weekends"] = [d for d in ['Saturday', 'Sunday'] if (d == 'Saturday' and 5 in missing_weekdays) or (d == 'Sunday' and 6 in missing_weekdays)]

            # 3. Check duplicated dates
            dup_count = df_check['record_date'].duplicated().sum()
            log["duplicated_dates"] = int(dup_count)

            # 4. Check format lỗi ngày
            invalid_dates = df_check['record_date'].isna().sum()
            log["invalid_date_format"] = int(invalid_dates)

            logs[ticker] = log
    else:
        raise ValueError("Dữ liệu không ở dạng dictionary sau fill_missing.")

    return {"preprocessing_check": logs}
