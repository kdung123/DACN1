# ╔══════════════════════════════════════════════════════════════╗
# ║                    data_processing.py                       ║
# ║     Load → Làm sạch → Feature Engineering → Split → Scale  ║
# ╠══════════════════════════════════════════════════════════════╣
# ║  Thứ tự gọi hàm trong pipeline:                            ║
# ║    1. load_stock_data()     — đọc CSV từ ổ cứng            ║
# ║    2. preprocess()          — làm sạch, sort ngày          ║
# ║    3. feature_engineering() — tạo 30+ features kỹ thuật    ║
# ║    4. split_data()          — chia Train/Val/Test           ║
# ║    5. scale_data()          — MinMaxScaler                  ║
# ║    6. make_sequences()      — sliding window cho BiGRU      ║
# ╠══════════════════════════════════════════════════════════════╣
# ║  CHỐNG DATA LEAKAGE & LAGGED VARIABLE BIAS                 ║
# ║  ─────────────────────────────────────────                 ║
# ║  • close_lag = Close.shift(1) làm nguồn cho MỌI feature    ║
# ║  • SMA/EMA/RSI/BB/Return đều tính từ close_lag             ║
# ║  • OHLV ngày T-1 dùng shift(1)                             ║
# ║  • scaler.fit() CHỈ trên train — val/test chỉ transform    ║
# ║  • Split theo thứ tự thời gian — KHÔNG shuffle             ║
# ╚══════════════════════════════════════════════════════════════╝

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

import config


# ════════════════════════════════════════════════════════════════
#  BƯỚC 1 — LOAD
#  Đọc file CSV từ thư mục data/VN30_Data/<TICKER>/
# ════════════════════════════════════════════════════════════════

def get_csv_path(ticker: str, mode: str = config.DATA_MODE) -> str:
    """Trả về đường dẫn đầy đủ đến file CSV của mã cổ phiếu."""
    import os
    return os.path.join(config.DATA_DIR, ticker, f"{ticker}_{mode}.csv")


def load_stock_data(ticker: str, mode: str = config.DATA_MODE) -> pd.DataFrame:
    """
    Đọc file CSV lịch sử giá của một mã cổ phiếu.

    Input  : data/VN30_Data/<TICKER>/<TICKER>_<mode>.csv
    Output : pd.DataFrame  cột gốc Date, Open, High, Low, Close, Volume

    Parameters
    ----------
    ticker : str   Mã cổ phiếu (vd: "VCB")
    mode   : str   "2y" | "5y" | "historical"

    Raises
    ------
    FileNotFoundError  Nếu file không tồn tại
    """
    import os
    path = get_csv_path(ticker, mode)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Không tìm thấy file: {path}")
    df = pd.read_csv(path)
    df["ticker"] = ticker
    return df


# ════════════════════════════════════════════════════════════════
#  BƯỚC 2 — TIỀN XỬ LÝ
#  Làm sạch dữ liệu thô trước khi tạo features
# ════════════════════════════════════════════════════════════════

def preprocess(df: pd.DataFrame, ticker: str = "") -> pd.DataFrame:
    """
    Làm sạch DataFrame dữ liệu giá thô.

    Các bước theo thứ tự
    ────────────────────
    1. Parse cột Date → datetime, sắp xếp tăng dần theo thời gian
    2. Thay giá 0 → NaN (giá 0 là lỗi dữ liệu, không phải giá thật)
    3. Forward-fill rồi back-fill NaN (giữ nguyên trend, không tạo thông tin mới)
    4. Xóa dòng thiếu Close (target — không thể đoán)
    5. Xóa ngày trùng lặp (giữ lần đầu xuất hiện)
    """
    df = df.copy()

    # ── 1. Parse & sort ngày ─────────────────────────────────
    df[config.DATE_COLUMN] = pd.to_datetime(df[config.DATE_COLUMN])
    df = df.sort_values(config.DATE_COLUMN).reset_index(drop=True)

    # ── 2–3. Xử lý giá 0 và NaN ──────────────────────────────
    existing = [c for c in config.OHLCV_COLS if c in df.columns]
    df[existing] = df[existing].replace(0, np.nan)
    df[existing] = df[existing].ffill().bfill()

    # ── 4–5. Xóa dòng không hợp lệ ───────────────────────────
    df = df.dropna(subset=[config.TARGET_COLUMN]).reset_index(drop=True)
    df = df.drop_duplicates(subset=[config.DATE_COLUMN]).reset_index(drop=True)

    print(f"  [{ticker}] {len(df)} phiên  "
          f"({df[config.DATE_COLUMN].min().date()} "
          f"→ {df[config.DATE_COLUMN].max().date()})")
    return df


# ════════════════════════════════════════════════════════════════
#  BƯỚC 3 — FEATURE ENGINEERING
#  Tạo đặc trưng kỹ thuật — tất cả tính từ close_lag = Close.shift(1)
#
#  Lý do dùng close_lag thay vì Close trực tiếp:
#    Close(T) là target cần dự báo — nếu dùng Close(T) để tính features,
#    model biết trước kết quả → circular dependency → metric ảo tốt
#    nhưng thực tế không dùng được.
# ════════════════════════════════════════════════════════════════

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tạo ~30 đặc trưng kỹ thuật, tất cả tính từ close_lag = Close.shift(1).

    Tại ngày T:
      close_lag(T) = Close(T-1)   ← model chỉ thấy giá đến hôm qua

    Nhóm features
    ─────────────
    Lag       : Lag_1..10   — Close(T-1) đến Close(T-10)
    Trend     : SMA_5..50   — trung bình động đơn giản
                EMA_5..50   — trung bình động lũy thừa
    Momentum  : RSI         — Relative Strength Index (14 ngày)
    Volatility: BB_Upper/Lower/Width/Pct — Bollinger Bands
                Return_1d/5d             — % thay đổi giá
                Volatility               — std của Return_1d (10 ngày)
    OHLV T-1  : HL_Ratio_prev — (High-Low)/Close ngày T-1
                OC_Ratio_prev — (Close-Open)/Open ngày T-1
                Vol_Ratio     — Volume(T-1)/VolMA5(T-1)
    """
    df        = df.copy()
    target    = config.TARGET_COLUMN

    # Nguồn gốc của TẤT CẢ features: Close đã shift 1 ngày
    # → Tại ngày T, close_lag = Close(T-1) — không có thông tin của T
    close_lag = df[target].shift(1)

    # ── Nhóm Lag ─────────────────────────────────────────────
    # Lag_1 = Close(T-1) = close_lag
    # Lag_k = Close(T-k) với k ∈ LAG_WINDOWS
    for lag in config.LAG_WINDOWS:
        df[f"Lag_{lag}"] = df[target].shift(lag)

    # ── Nhóm Trend — tính từ close_lag ───────────────────────
    # SMA_20(T) = mean(Close[T-20:T-1]) — window 20 ngày trước T
    # EMA dùng ewm() — trung bình lũy thừa có trọng số giảm dần
    for w in config.MA_WINDOWS:
        df[f"SMA_{w}"] = close_lag.rolling(w).mean()
        df[f"EMA_{w}"] = close_lag.ewm(span=w, adjust=False).mean()

    # ── RSI 14 — tính từ close_lag ───────────────────────────
    # delta(T) = Close(T-1) - Close(T-2)   → không có Close(T)
    # gain = mean(delta > 0) trong 14 ngày
    # loss = mean(delta < 0) trong 14 ngày (lấy abs)
    # RSI = 100 - 100/(1 + gain/loss)
    delta     = close_lag.diff()
    gain      = delta.clip(lower=0).rolling(config.RSI_PERIOD).mean()
    loss      = (-delta.clip(upper=0)).rolling(config.RSI_PERIOD).mean()
    df["RSI"] = 100 - (100 / (1 + gain / (loss + 1e-9)))

    # ── Bollinger Bands — tính từ close_lag ──────────────────
    # BB_Upper(T) = SMA_20(T) + 2×std_20(T)
    # Vì dùng close_lag, window = Close[T-20:T-1] — không có Close(T)
    mid            = close_lag.rolling(config.BB_PERIOD).mean()
    std            = close_lag.rolling(config.BB_PERIOD).std()
    df["BB_Upper"] = mid + config.BB_STD * std
    df["BB_Lower"] = mid - config.BB_STD * std
    df["BB_Width"] = df["BB_Upper"] - df["BB_Lower"]
    df["BB_Pct"]   = (close_lag - df["BB_Lower"]) / (df["BB_Width"] + 1e-9)
    # BB_Pct: 0 = đang ở đáy dải, 1 = đang ở đỉnh dải

    # ── Returns & Volatility — tính từ close_lag ─────────────
    # Return_1d(T) = (Close(T-1) - Close(T-2)) / Close(T-2)
    # Volatility(T) = std của Return_1d trong 10 ngày gần nhất
    df["Return_1d"]  = close_lag.pct_change(1)
    df["Return_5d"]  = close_lag.pct_change(5)
    df["Volatility"] = df["Return_1d"].rolling(10).std()

    # ── OHLV ngày T-1 — shift(1) để tránh leak ───────────────
    # HL_Ratio_prev: biên độ dao động ngày hôm qua so với giá đóng cửa
    # OC_Ratio_prev: cây nến (bullish/bearish) ngày hôm qua
    if all(c in df.columns for c in ["Open", "High", "Low"]):
        df["HL_Ratio_prev"] = (
            (df["High"].shift(1) - df["Low"].shift(1))
            / (close_lag + 1e-9)
        )
        df["OC_Ratio_prev"] = (
            (close_lag - df["Open"].shift(1))
            / (df["Open"].shift(1) + 1e-9)
        )

    # Vol_Ratio: khối lượng hôm qua so với trung bình 5 ngày
    # → phát hiện ngày có khối lượng bất thường (đột biến)
    if "Volume" in df.columns:
        vol_lag         = df["Volume"].shift(1)
        vol_ma5         = vol_lag.rolling(5).mean()
        df["Vol_Ratio"] = vol_lag / (vol_ma5 + 1e-9)

    # ── Xóa cột OHLCV gốc ────────────────────────────────────
    # Model không được thấy Open/High/Low/Volume của ngày T
    # (chỉ thấy các giá trị đã shift của ngày T-1)
    drop = [c for c in ["Open", "High", "Low", "Volume", "ticker"]
            if c in df.columns]
    df = df.drop(columns=drop)

    # ── Xóa NaN đầu do rolling/shift ─────────────────────────
    # Các rolling window lớn nhất (SMA_50, BB_20) tạo NaN ở 50 dòng đầu
    before = len(df)
    df     = df.dropna().reset_index(drop=True)
    print(f"  Feature engineering: {df.shape[1]} cột, {len(df)} dòng  "
          f"(loại {before - len(df)} dòng NaN đầu)")
    return df


def get_feature_columns(df: pd.DataFrame) -> list:
    """Trả về danh sách cột feature (bỏ Date, ticker)."""
    exclude = {config.DATE_COLUMN, "ticker"}
    return [c for c in df.columns if c not in exclude]


# ════════════════════════════════════════════════════════════════
#  BƯỚC 4 — SPLIT
#  Chia theo thứ tự thời gian — KHÔNG shuffle
#
#  Lý do không shuffle:
#    Cổ phiếu là chuỗi thời gian — dữ liệu tương lai không được
#    lọt vào tập train. Shuffle phá vỡ thứ tự → data leakage.
# ════════════════════════════════════════════════════════════════

def split_data(df: pd.DataFrame):
    """
    Chia dữ liệu theo tỷ lệ TRAIN 70 / VAL 15 / TEST 15.
    Thứ tự: Train (cũ nhất) → Val → Test (mới nhất).

    Returns
    -------
    (train, val, test) : tuple of pd.DataFrame
    """
    n       = len(df)
    n_train = int(n * config.TRAIN_RATIO)
    n_val   = int(n * config.VAL_RATIO)

    train = df.iloc[:n_train].copy()
    val   = df.iloc[n_train : n_train + n_val].copy()
    test  = df.iloc[n_train + n_val :].copy()

    print(f"  Split → Train: {len(train)}  Val: {len(val)}  Test: {len(test)}")
    return train, val, test


# ════════════════════════════════════════════════════════════════
#  BƯỚC 5 — SCALE
#  MinMaxScaler cho features (X) và target (y)
#
#  Quy tắc bắt buộc:
#    scaler.fit() CHỈ trên train — không dùng thông tin val/test
#    scaler.transform() cho val và test — chỉ áp dụng scale đã học
#
#  Lý do:
#    Fit scaler trên val/test = model biết phân phối dữ liệu tương lai
#    → data leakage → metric ảo tốt hơn thực tế
# ════════════════════════════════════════════════════════════════

def scale_data(train, val, test, features):
    """
    Chuẩn hóa X và y về [0, 1] bằng MinMaxScaler.

    Parameters
    ----------
    features : list  Tất cả cột feature (gồm cả target Close)

    Returns
    -------
    X_train, y_train, X_val, y_val, X_test, y_test, scaler_X, scaler_y
      Trong đó scaler_X và scaler_y chỉ được fit trên train.
    """
    target = config.TARGET_COLUMN
    X_cols = [f for f in features if f != target]

    # Fit ONLY trên train
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X_train = scaler_X.fit_transform(train[X_cols])
    y_train = scaler_y.fit_transform(train[[target]]).ravel()

    # Val và Test — chỉ transform, không fit lại
    X_val  = scaler_X.transform(val[X_cols])
    y_val  = scaler_y.transform(val[[target]]).ravel()
    X_test = scaler_X.transform(test[X_cols])
    y_test = scaler_y.transform(test[[target]]).ravel()

    return X_train, y_train, X_val, y_val, X_test, y_test, scaler_X, scaler_y


# ════════════════════════════════════════════════════════════════
#  BƯỚC 6 — SLIDING WINDOW SEQUENCES (chỉ dùng cho BiGRU)
#  Chuyển array 2D → array 3D (samples, timesteps, features)
#
#  Với mỗi vị trí i:
#    X_seq[i] = X[i : i+seq_len]   — seq_len phiên liên tiếp
#    y_seq[i] = y[i + seq_len]     — phiên cần dự báo
#
#  Hệ quả quan trọng:
#    Output ngắn hơn input đúng seq_len phiên
#    → BiGRU bắt đầu predict muộn hơn seq_len ngày so với Ridge/RF
# ════════════════════════════════════════════════════════════════

def make_sequences(X: np.ndarray, y: np.ndarray,
                   seq_len: int = config.SEQUENCE_LEN):
    """
    Tạo sliding-window sequences cho BiGRU.

    Parameters
    ----------
    X       : np.ndarray  shape (n_days, n_features)
    y       : np.ndarray  shape (n_days,)
    seq_len : int         độ dài cửa sổ (ngày)

    Returns
    -------
    Xs : np.ndarray  shape (n_samples, seq_len, n_features)
    ys : np.ndarray  shape (n_samples,)
    """
    Xs, ys = [], []
    for i in range(len(X) - seq_len):
        Xs.append(X[i : i + seq_len])
        ys.append(y[i + seq_len])
    return np.array(Xs, dtype=np.float32), np.array(ys, dtype=np.float32)
