import os
import pandas as pd

csv_path = "/Users/minhtan/Documents/GitHub/Time_Series_Forecast/AI_Agent/output/investment_results/investment_decision.csv"

if not os.path.exists(csv_path):
    print("[ERROR] File investment_decision.csv không tồn tại.")

df = pd.read_csv(csv_path)
if "strategy" not in df.columns or df.empty:
    print("[ERROR] Không có nội dung chiến lược để gửi.")

strategy_text = str(df["strategy"].iloc[-1]).strip()

if not strategy_text:
    print("[ERROR] Chiến lược rỗng.")

print(strategy_text)