import os
import pandas as pd

def generate_summary(results_dir, output_path="summary_all_tickers.csv"):
    summary = {}

    for file in os.listdir(results_dir):
        if file.endswith("_gru_results.csv"):
            ticker = file.split("_")[0]
            file_path = os.path.join(results_dir, file)

            try:
                df = pd.read_csv(file_path)

                # Chỉ giữ các dòng hợp lệ (có đầy đủ MAE, RMSE, MAPE)
                if not all(col in df.columns for col in ["MAE", "RMSE", "MAPE"]):
                    print(f"[WARN] {ticker}: Thiếu cột MAE/RMSE/MAPE")
                    continue

                df_valid = df[["MAE", "RMSE", "MAPE"]].dropna(how="any")
                if df_valid.empty:
                    print(f"[WARN] {ticker}: Không có dòng hợp lệ")
                    continue

                # Tính trung bình
                avg_metrics = df_valid.mean().to_dict()
                avg_metrics["ticker"] = ticker
                avg_metrics["result_path"] = file_path
                summary[ticker] = avg_metrics

                print(f"[INFO] Done: {ticker}")

            except Exception as e:
                print(f"[ERROR] {ticker}: Lỗi khi đọc hoặc xử lý file - {e}")
                continue

    if summary:
        df_summary = pd.DataFrame.from_dict(summary, orient="index")
        df_summary.to_csv(os.path.join(results_dir, output_path), index=False)
        print(f"[SUCCESS] Đã ghi file summary tại: {os.path.join(results_dir, output_path)}")
    else:
        print("[ERROR] Không có dữ liệu nào hợp lệ để ghi summary.")

# Ví dụ cách chạy script
if __name__ == "__main__":
    results_dir = "/Users/minhtan/Documents/GitHub/Time_Series_Forecast/AI_Agent/output/output_models/gru_results"
    generate_summary(results_dir)
