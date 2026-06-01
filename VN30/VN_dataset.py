import os
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from dotenv import load_dotenv
from tqdm import tqdm
import logging


YEARS_TO_GET = 5  

# Cấu hình Logging để theo dõi tiến trình
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

class VN30Downloader:
    def __init__(self, years: int):
        load_dotenv()
        self.years = years
        # Lấy cấu hình từ .env hoặc mặc định
        self.output_dir = Path(os.getenv("BASE_DATA_DIR", "VN30_Data"))
        self.tickers = [t.strip() for t in os.getenv("VN30_TICKERS", "").split(",") if t.strip()]
        
        # Tự động tính khoảng thời gian
        self.end_date = datetime.now()
        self.start_date = self.end_date - timedelta(days=365 * self.years)
        
    def prepare_folders(self):
        """Tạo cấu trúc thư mục sạch sẽ."""
        for ticker in self.tickers:
            (self.output_dir / ticker).mkdir(parents=True, exist_ok=True)
        logger.info(f"📁 Đã cấu trúc thư mục tại: {self.output_dir}")

    def download_all(self):
        """Tải và phân loại dữ liệu đa luồng."""
        yahoo_list = [f"{t}.VN" for t in self.tickers]
        
        logger.info(f"🚀 Đang tải {self.years} năm dữ liệu cho {len(self.tickers)} mã VN30...")
        
        # Tải gộp một lần duy nhất để tối ưu tốc độ
        full_data = yf.download(
            tickers=yahoo_list,
            start=self.start_date.strftime('%Y-%m-%d'),
            end=self.end_date.strftime('%Y-%m-%d'),
            group_by='ticker',
            threads=True,
            progress=False
        )

        # Lưu trữ từng mã vào folder tương ứng
        for ticker in tqdm(self.tickers, desc="Đang lưu file CSV"):
            ticker_vn = f"{ticker}.VN"
            if ticker_vn in full_data.columns.get_level_values(0):
                df = full_data[ticker_vn].dropna().reset_index()
                
                if not df.empty:
                    file_path = self.output_dir / ticker / f"{ticker}_{self.years}y.csv"
                    df.to_csv(file_path, index=False)
                    
    def start(self):
        """Quy trình thực thi chính."""
        if not self.tickers:
            logger.error("❌ Không tìm thấy danh sách mã trong .env!")
            return
        self.prepare_folders()
        self.download_all()
        logger.info(f"✨ HOÀN THÀNH! Dữ liệu đã sẵn sàng để làm AI.")

if __name__ == "__main__":
    # Khởi tạo và chạy với số năm đã chọn ở đầu file
    downloader = VN30Downloader(years=YEARS_TO_GET)
    downloader.start()