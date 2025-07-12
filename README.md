# Time Series Forecast
Toàn bộ source code của môn học Phân tích chuỗi thời gian và dự báo (Time Series Analysis and Forecast) do thầy Lê Hoành Sử hướng dẫn. Cấu trúc dự án bao gồm:
* AI_Agent: code triển khai đầy đủ toàn bộ quy trình phân tích sử dụng AI Agent (framework LangGraph)

* Traditional_Code: code triển khai phân tích sử dụng quy trình truyền thống thuần python

Cấu trúc thư mục AI_Agent bao gồm:
* agent: chứa file main
* config: cấu hình dự án và các dữ liệu cá nhân
* data_update_daily: dữ liệu được cập nhật mỗi ngày theo yfinance
* logs: ghi lại log
* output: đầu ra kết quả của từng node thay cho memory
* tests: chứa file test của dự án
* tools: toàn bộ các node được xây dựng thành từng file python trong này

Cấu trúc thư mục Traditional_Code bao gồm:
* EDA: chứa dữ liệu data đã được EDA hoàn chỉnh
* model: xây dựng model