# ============================================================
# config.py - Cấu hình dự án dự báo giá cổ phiếu VN30
# ============================================================

import os

# --- Đường dẫn dữ liệu thực tế ---
VN30_DIR     = "VN30"                      # Thư mục gốc chứa dữ liệu crawl
DATA_DIR     = os.path.join(VN30_DIR, "VN30_Data")  # VN30/VN30_Data/<TICKER>/
RESULTS_DIR  = "results"
MODELS_DIR   = "models"

# --- Loại file dữ liệu: "2y" | "5y" | "historical" ---
# "historical" = 5 năm (dữ liệu đầy đủ nhất, dùng mặc định)
DATA_MODE = "historical"

# --- Danh sách 30 mã VN30 ---
VN30_TICKERS = [
    "ACB","BCM","BID","BVH","CTG","FPT","GAS","GVR","HDB","HPG",
    "MBB","MSN","MWG","PLX","POW","REE","SAB","SHB","SSB","SSI",
    "STB","TCB","TPB","VCB","VHM","VIB","VIC","VJC","VNM","VPB",
]

# --- Cấu hình cột (khớp đúng với CSV crawl) ---
DATE_COLUMN   = "Date"    # Cột ngày
TARGET_COLUMN = "Close"   # Cột giá dự báo
OHLCV_COLS    = ["Open", "High", "Low", "Close", "Volume"]

# --- Tỷ lệ chia dữ liệu ---
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15

# --- Feature Engineering ---
LAG_WINDOWS = [1, 2, 3, 5, 10]
MA_WINDOWS  = [5, 10, 20, 50]
RSI_PERIOD  = 14
BB_PERIOD   = 20
BB_STD      = 2

# --- ARIMA ---
ARIMA_ORDER = (5, 1, 0)   # (p, d, q)

# --- Machine Learning ---
RF_N_ESTIMATORS = 200
RF_RANDOM_STATE = 42

# --- Deep Learning ---
SEQUENCE_LEN   = 30       # Sliding window cho LSTM/GRU
LSTM_UNITS     = [64, 32]
GRU_UNITS      = [64, 32]
DROPOUT_RATE   = 0.2
DL_EPOCHS      = 50
DL_BATCH_SIZE  = 32
DL_PATIENCE    = 10

RANDOM_SEED = 42
