# ╔══════════════════════════════════════════════════════════════╗
# ║                    data_processing.py                       ║
# ║        Load · Tiền xử lý · Feature Engineering             ║
# ╚══════════════════════════════════════════════════════════════╝
#
# NGUYÊN TẮC CHỐNG DATA LEAKAGE
# ─────────────────────────────
#  • Open / High / Low  → chỉ dùng qua .shift(1)  (ngày T-1)
#  • Volume             → chỉ dùng qua .shift(1)  (ngày T-1)
#  • Bollinger Bands    → tính từ Close.shift(1)   (không gồm Close T)
#  • scaler_X / scaler_y → .fit_transform() CHỈ trên tập TRAIN
#  • Split              → theo thứ tự thời gian, KHÔNG shuffle

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

import config


# ════════════════════════════════════════════════════════════════
#  1. LOAD
# ════════════════════════════════════════════════════════════════

def get_csv_path(ticker: str, mode: str = config.DATA_MODE) -> str:
    """Trả về đường dẫn đến file CSV của mã cổ phiếu."""
    return os.path.join(config.DATA_DIR, ticker, f"{ticker}_{mode}.csv")


def load_stock_data(ticker: str, mode: str = config.DATA_MODE) -> pd.DataFrame:
    """
    Đọc file CSV lịch sử giá của một mã cổ phiếu.

    Parameters
    ----------
    ticker : str   Mã cổ phiếu (vd: "VCB")
    mode   : str   Loại file — "2y" | "5y" | "historical"

    Returns
    -------
    pd.DataFrame   Cột: Date, Open, High, Low, Close, Volume
    """
    path = get_csv_path(ticker, mode)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Không tìm thấy file: {path}")
    df = pd.read_csv(path)
    df["ticker"] = ticker
    return df


def load_all_tickers(mode: str = config.DATA_MODE, tickers: list = None) -> dict:
    """Đọc tất cả mã trong danh sách, bỏ qua mã bị lỗi."""
    tickers = tickers or config.VN30_TICKERS
    data    = {}
    for t in tickers:
        try:
            data[t] = load_stock_data(t, mode)
        except FileNotFoundError as e:
            print(f"  ⚠  Bỏ qua {t}: {e}")
    return data


# ════════════════════════════════════════════════════════════════
#  2. TIỀN XỬ LÝ
# ════════════════════════════════════════════════════════════════

def preprocess(df: pd.DataFrame, ticker: str = "") -> pd.DataFrame:
    """
    Làm sạch dữ liệu thô:
      1. Parse ngày, sắp xếp tăng dần
      2. Thay 0 → NaN, forward-fill / back-fill giá trị thiếu
      3. Xóa dòng thiếu Close, xóa ngày trùng lặp
    """
    df = df.copy()

    # 1. Parse & sort
    df[config.DATE_COLUMN] = pd.to_datetime(df[config.DATE_COLUMN])
    df = df.sort_values(config.DATE_COLUMN).reset_index(drop=True)

    # 2. Xử lý giá trị 0 / NaN
    existing = [c for c in config.OHLCV_COLS if c in df.columns]
    df[existing] = df[existing].replace(0, np.nan)
    df[existing] = df[existing].ffill().bfill()

    # 3. Xóa dòng không hợp lệ
    df = df.dropna(subset=[config.TARGET_COLUMN]).reset_index(drop=True)
    df = df.drop_duplicates(subset=[config.DATE_COLUMN]).reset_index(drop=True)

    print(f"  [{ticker}] {len(df)} phiên  "
          f"({df[config.DATE_COLUMN].min().date()} → {df[config.DATE_COLUMN].max().date()})")
    return df


# ════════════════════════════════════════════════════════════════
#  3. FEATURE ENGINEERING
# ════════════════════════════════════════════════════════════════

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tạo các đặc trưng kỹ thuật từ dữ liệu giá.

    Tất cả features đều được tính sao cho tại ngày T,
    model CHỈ thấy thông tin từ T-1 trở về trước.

    Features tạo ra
    ───────────────
    Lag_1..10      : Giá đóng cửa T-1 đến T-10
    SMA_5..50      : Simple Moving Average
    EMA_5..50      : Exponential Moving Average
    RSI            : Relative Strength Index (14)
    BB_Upper/Lower : Bollinger Bands (tính từ Close.shift(1))
    BB_Width       : Độ rộng Bollinger Bands
    BB_Pct         : Vị trí giá trong BB (0=đáy, 1=đỉnh)
    Return_1d/5d   : % thay đổi giá 1 ngày / 5 ngày
    Volatility     : Độ lệch chuẩn Return_1d (10 ngày)
    HL_Ratio_prev  : (High-Low) / Close của ngày T-1
    OC_Ratio_prev  : (Close-Open) / Open của ngày T-1
    Vol_Ratio      : Volume(T-1) / VolMA5(T-1)
    """
    df     = df.copy()
    target = config.TARGET_COLUMN

    # ── Lag features ─────────────────────────────────────────
    for lag in config.LAG_WINDOWS:
        df[f"Lag_{lag}"] = df[target].shift(lag)

    # ── Moving Averages ───────────────────────────────────────
    for w in config.MA_WINDOWS:
        df[f"SMA_{w}"] = df[target].rolling(w).mean()
        df[f"EMA_{w}"] = df[target].ewm(span=w, adjust=False).mean()

    # ── RSI (14) ─────────────────────────────────────────────
    delta        = df[target].diff()
    gain         = delta.clip(lower=0).rolling(config.RSI_PERIOD).mean()
    loss         = (-delta.clip(upper=0)).rolling(config.RSI_PERIOD).mean()
    df["RSI"]    = 100 - (100 / (1 + gain / (loss + 1e-9)))

    # ── Bollinger Bands — shift(1) tránh leak Close(T) ───────
    close_lag    = df[target].shift(1)
    mid          = close_lag.rolling(config.BB_PERIOD).mean()
    std          = close_lag.rolling(config.BB_PERIOD).std()
    df["BB_Upper"] = mid + config.BB_STD * std
    df["BB_Lower"] = mid - config.BB_STD * std
    df["BB_Width"] = df["BB_Upper"] - df["BB_Lower"]
    df["BB_Pct"]   = (close_lag - df["BB_Lower"]) / (df["BB_Width"] + 1e-9)

    # ── Returns & Volatility ──────────────────────────────────
    df["Return_1d"]  = df[target].pct_change(1)
    df["Return_5d"]  = df[target].pct_change(5)
    df["Volatility"] = df["Return_1d"].rolling(10).std()

    # ── OHLC ngày T-1 (shift = không leak) ───────────────────
    if all(c in df.columns for c in ["Open", "High", "Low"]):
        df["HL_Ratio_prev"] = (
            (df["High"].shift(1) - df["Low"].shift(1))
            / (df[target].shift(1) + 1e-9)
        )
        df["OC_Ratio_prev"] = (
            (df[target].shift(1) - df["Open"].shift(1))
            / (df["Open"].shift(1) + 1e-9)
        )

    # ── Volume ngày T-1 ───────────────────────────────────────
    if "Volume" in df.columns:
        df["Vol_MA5"]  = df["Volume"].rolling(5).mean()
        df["Vol_Ratio"] = df["Volume"].shift(1) / (df["Vol_MA5"].shift(1) + 1e-9)

    # ── Xóa cột gốc — model không thấy OHLCV hiện tại ────────
    drop_cols = [c for c in ["Open", "High", "Low", "Volume", "ticker"]
                 if c in df.columns]
    df = df.drop(columns=drop_cols)

    # ── Xóa dòng NaN do rolling / shift ──────────────────────
    before = len(df)
    df     = df.dropna().reset_index(drop=True)
    print(f"  Feature engineering: {df.shape[1]} cột, {len(df)} dòng  "
          f"(loại {before - len(df)} dòng NaN đầu)")
    return df


# ════════════════════════════════════════════════════════════════
#  4. SPLIT · SCALE · SEQUENCE
# ════════════════════════════════════════════════════════════════

def split_data(df: pd.DataFrame):
    """
    Chia dữ liệu theo tỷ lệ TRAIN / VAL / TEST theo thứ tự thời gian.
    KHÔNG shuffle để tránh data leakage.
    """
    n       = len(df)
    n_train = int(n * config.TRAIN_RATIO)
    n_val   = int(n * config.VAL_RATIO)

    train = df.iloc[:n_train].copy()
    val   = df.iloc[n_train : n_train + n_val].copy()
    test  = df.iloc[n_train + n_val :].copy()

    print(f"  Split → Train: {len(train)}  Val: {len(val)}  Test: {len(test)}")
    return train, val, test


def get_feature_columns(df: pd.DataFrame) -> list:
    """Trả về danh sách cột dùng làm feature (bỏ Date, ticker)."""
    exclude = {config.DATE_COLUMN, "ticker"}
    return [c for c in df.columns if c not in exclude]


def scale_data(train, val, test, features):
    """
    MinMaxScaler cho X và y.
    scaler.fit() CHỈ trên tập train — val/test chỉ được transform.

    Returns
    -------
    X_train, y_train, X_val, y_val, X_test, y_test, scaler_X, scaler_y
    """
    target  = config.TARGET_COLUMN
    X_feats = [f for f in features if f != target]

    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X_train = scaler_X.fit_transform(train[X_feats])
    X_val   = scaler_X.transform(val[X_feats])
    X_test  = scaler_X.transform(test[X_feats])

    y_train = scaler_y.fit_transform(train[[target]]).ravel()
    y_val   = scaler_y.transform(val[[target]]).ravel()
    y_test  = scaler_y.transform(test[[target]]).ravel()

    return X_train, y_train, X_val, y_val, X_test, y_test, scaler_X, scaler_y


def make_sequences(X: np.ndarray, y: np.ndarray,
                   seq_len: int = config.SEQUENCE_LEN):
    """
    Tạo sliding-window sequences cho BiGRU.

    Với mỗi vị trí i:
      X_seq[i] = X[i : i+seq_len]     shape: (seq_len, n_features)
      y_seq[i] = y[i + seq_len]        giá trị cần dự báo

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
