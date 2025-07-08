import os
from dotenv import load_dotenv

# Tải các biến từ config.env
# Cách tổng quát

'''
current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root_dir = os.path.dirname(current_script_dir)
config_env_path = os.path.join(project_root_dir, 'config', 'config.env')

# Tải biến môi trường
load_dotenv(dotenv_path=config_env_path)
'''

# Cách cực đoan
load_dotenv(dotenv_path='/Users/minhtan/Documents/GitHub/Time-Series-Forecast/AI-Agent/config/config.env')

# Truy cập các biến môi trường
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
ALPHAVANTAGE_API_KEY = os.getenv("ALPHAVANTAGE_API_KEY")

# Kiểm tra xem các biến đã được tải chưa
if GOOGLE_API_KEY:
    print(f"Google API Key: {GOOGLE_API_KEY}")
else:
    print("Không tìm thấy Google API Key. Hãy kiểm tra file của bạn.")

if ALPHAVANTAGE_API_KEY:
    print(f"ALPHAVANTAGE API Key: {ALPHAVANTAGE_API_KEY}")
else:
    print("Không tìm thấy ALPHAVANTAGE API Key. Hãy kiểm tra file của bạn.")