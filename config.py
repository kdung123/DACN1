# ╔══════════════════════════════════════════════════════════════╗
# ║                        config.py                            ║
# ║              Cấu hình toàn cục — VN30 Project              ║
# ╠══════════════════════════════════════════════════════════════╣
# ║  File này được import đầu tiên bởi tất cả các module.      ║
# ║  Mọi tham số cần chỉnh đều nằm ở đây — không hardcode      ║
# ║  rải rác trong code.                                        ║
# ╚══════════════════════════════════════════════════════════════╝

import os

# ════════════════════════════════════════════════════════════════
#  BƯỚC 0 — Đường dẫn thư mục
#  Tất cả path tính tương đối từ vị trí file này
# ════════════════════════════════════════════════════════════════

VN30_DIR    = "VN30"
DATA_DIR    = os.path.join(VN30_DIR, "VN30_Data")   # VN30/VN30_Data/<TICKER>/
RESULTS_DIR = "results"                              # lưu biểu đồ & CSV kết quả
MODELS_DIR  = "models"                               # lưu file .keras đã train

# Loại file dữ liệu mặc định khi train
# Các giá trị hợp lệ: "2y" | "5y" | "historical"
DATA_MODE = "historical"


# ════════════════════════════════════════════════════════════════
#  BƯỚC 1 — Danh sách 30 mã VN30
#  Dùng để crawl dữ liệu và chạy batch train toàn bộ rổ
# ════════════════════════════════════════════════════════════════

VN30_TICKERS = [
    "ACB", "BCM", "BID", "BVH", "CTG", "FPT", "GAS", "GVR", "HDB", "HPG",
    "MBB", "MSN", "MWG", "PLX", "POW", "REE", "SAB", "SHB", "SSB", "SSI",
    "STB", "TCB", "TPB", "VCB", "VHM", "VIB", "VIC", "VJC", "VNM", "VPB",
]


# ════════════════════════════════════════════════════════════════
#  BƯỚC 2 — Cấu hình cột dữ liệu
#  Phải khớp với tên cột trong file CSV đầu vào
# ════════════════════════════════════════════════════════════════

DATE_COLUMN   = "Date"
TARGET_COLUMN = "Target"                                    # cột cần dự báo
OHLCV_COLS    = ["Open", "High", "Low", "Close", "Volume"] # tất cả cột giá


# ════════════════════════════════════════════════════════════════
#  BƯỚC 3 — Tỷ lệ chia dữ liệu
#  Chia theo THỨ TỰ THỜI GIAN — không shuffle
#  Train (cũ nhất) → Val → Test (mới nhất)
# ════════════════════════════════════════════════════════════════

TRAIN_RATIO = 0.70   # 70% — huấn luyện model
VAL_RATIO   = 0.15   # 15% — tune hyperparameter & tìm ensemble weights
TEST_RATIO  = 0.15   # 15% — đánh giá cuối cùng (chỉ dùng 1 lần)


# ════════════════════════════════════════════════════════════════
#  BƯỚC 4 — Feature Engineering
#  Tất cả features đều tính từ Close.shift(1) hoặc shift(N)
#  → Không có thông tin ngày T trong input của model
# ════════════════════════════════════════════════════════════════

# Lag features: Close của N ngày trước
LAG_WINDOWS = [3, 5, 10]

# Moving average windows (SMA & EMA)
MA_WINDOWS  = [5, 10, 20, 50]

# RSI lookback period (ngày)
RSI_PERIOD  = 14

# Bollinger Bands
BB_PERIOD   = 20    # lookback window
BB_STD      = 2     # số độ lệch chuẩn


# ════════════════════════════════════════════════════════════════
#  BƯỚC 5 — Cấu hình Optuna (tìm hyperparameter)
#  Số trials càng nhiều → tìm được tham số tốt hơn
#  nhưng tốn thời gian hơn tuyến tính
# ════════════════════════════════════════════════════════════════

RIDGE_TRIALS    = 30    # Ridge: 1 chiều (alpha) → 30 là đủ
RF_TRIALS       = 40    # RF: 4 chiều → cần nhiều hơn
ENSEMBLE_TRIALS = 100   # Ensemble: N model × trọng số → cần nhiều nhất


# ════════════════════════════════════════════════════════════════
#  BƯỚC 6 — Cấu hình BiGRU (Deep Learning)
#
#  SEQUENCE_LEN: số phiên nhìn về quá khứ để dự báo 1 phiên
#    → Khuyến nghị: 20–30 phiên (~1–1.5 tháng giao dịch)
#    → KHÔNG dùng > 60: ít training samples hơn, dễ overfit
#
#  DL_PATIENCE: EarlyStopping dừng sau bao nhiêu epoch không cải thiện
#    → ~10% tổng epochs là hợp lý
#    → Kết hợp ReduceLROnPlateau(patience=5) → LR giảm trước khi dừng
# ════════════════════════════════════════════════════════════════

SEQUENCE_LEN  = 30     # sliding window (phiên)
GRU_UNITS     = [64, 32]
DROPOUT_RATE  = 0.3    # dropout input weights (tăng từ 0.2 → giảm overfit)
DL_EPOCHS     = 100
DL_BATCH_SIZE = 32
DL_PATIENCE   = 10     # EarlyStopping patience

RANDOM_SEED   = 42
