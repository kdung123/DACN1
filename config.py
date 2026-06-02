# ╔══════════════════════════════════════════════════════════════╗
# ║                        config.py                            ║
# ║         Cấu hình toàn cục — VN30 Stock Prediction          ║
# ║  Models: Ridge (Optuna) · RF (Optuna) · BiGRU · Ensemble   ║
# ╚══════════════════════════════════════════════════════════════╝

import os

# ── Đường dẫn ────────────────────────────────────────────────────
VN30_DIR    = "VN30"
DATA_DIR    = os.path.join(VN30_DIR, "VN30_Data")   # VN30/VN30_Data/<TICKER>/
RESULTS_DIR = "results"
MODELS_DIR  = "models"
DATA_MODE   = "historical"                           # "2y" | "5y" | "historical"

# ── Danh sách 30 mã VN30 ─────────────────────────────────────────
VN30_TICKERS = [
    "ACB", "BCM", "BID", "BVH", "CTG", "FPT", "GAS", "GVR", "HDB", "HPG",
    "MBB", "MSN", "MWG", "PLX", "POW", "REE", "SAB", "SHB", "SSB", "SSI",
    "STB", "TCB", "TPB", "VCB", "VHM", "VIB", "VIC", "VJC", "VNM", "VPB",
]

# ── Cấu hình cột ────────────────────────────────────────────────
DATE_COLUMN   = "Date"
TARGET_COLUMN = "Close"
OHLCV_COLS    = ["Open", "High", "Low", "Close", "Volume"]

# ── Tỷ lệ chia dữ liệu ──────────────────────────────────────────
TRAIN_RATIO = 0.70   # 70% train
VAL_RATIO   = 0.15   # 15% validation
TEST_RATIO  = 0.15   # 15% test

# ── Feature Engineering ──────────────────────────────────────────
LAG_WINDOWS = [1, 2, 3, 5, 10]     # Lag Close (ngày)
MA_WINDOWS  = [5, 10, 20, 50]      # Cửa sổ SMA / EMA
RSI_PERIOD  = 14                   # RSI lookback
BB_PERIOD   = 20                   # Bollinger Bands lookback
BB_STD      = 2                    # Bollinger Bands số std

# ── Optuna ───────────────────────────────────────────────────────
RIDGE_TRIALS    = 30    # Ridge: không gian 1 chiều, 30 là đủ
RF_TRIALS       = 40    # RF: 4 tham số, cần nhiều hơn
ENSEMBLE_TRIALS = 100   # Ensemble: 4 model × trọng số liên tục

# ── BiGRU ────────────────────────────────────────────────────────
SEQUENCE_LEN  = 30    # Sliding window (khuyến nghị 20–30, KHÔNG dùng > 60)
GRU_UNITS     = [64, 32]
DROPOUT_RATE  = 0.3   # Tăng lên 0.3 để hạn chế overfitting
DL_EPOCHS     = 100
DL_BATCH_SIZE = 32
DL_PATIENCE   = 10    # EarlyStopping patience (~10% tổng epochs)

RANDOM_SEED = 42
