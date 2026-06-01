# ============================================================
# data_processing.py  –  Load, tiền xử lý, feature engineering
# FIX: loại bỏ data leakage (Open/High/Low của ngày hiện tại)
# ============================================================

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import config


def get_csv_path(ticker: str, mode: str = config.DATA_MODE) -> str:
    fname = f"{ticker}_{mode}.csv"
    return os.path.join(config.DATA_DIR, ticker, fname)


def load_stock_data(ticker: str, mode: str = config.DATA_MODE) -> pd.DataFrame:
    path = get_csv_path(ticker, mode)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Không tìm thấy: {path}")
    df = pd.read_csv(path)
    df["ticker"] = ticker
    return df


def load_all_tickers(mode: str = config.DATA_MODE, tickers: list = None) -> dict:
    tickers = tickers or config.VN30_TICKERS
    data = {}
    for t in tickers:
        try:
            data[t] = load_stock_data(t, mode)
        except FileNotFoundError as e:
            print(f"  ⚠ Bỏ qua {t}: {e}")
    return data


def preprocess(df: pd.DataFrame, ticker: str = "") -> pd.DataFrame:
    df = df.copy()
    df[config.DATE_COLUMN] = pd.to_datetime(df[config.DATE_COLUMN])
    df = df.sort_values(config.DATE_COLUMN).reset_index(drop=True)
    existing_ohlcv = [c for c in config.OHLCV_COLS if c in df.columns]
    df[existing_ohlcv] = df[existing_ohlcv].replace(0, np.nan)
    df[existing_ohlcv] = df[existing_ohlcv].ffill().bfill()
    df = df.dropna(subset=[config.TARGET_COLUMN]).reset_index(drop=True)
    df = df.drop_duplicates(subset=[config.DATE_COLUMN]).reset_index(drop=True)
    print(f"  [{ticker}] Sau tiền xử lý: {len(df)} phiên "
          f"({df[config.DATE_COLUMN].min().date()} → {df[config.DATE_COLUMN].max().date()})")
    return df


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    ĐÃ SỬA DATA LEAKAGE:
    - Không dùng Open/High/Low của ngày t (chưa biết khi dự báo Close(t))
    - Chỉ dùng Close lịch sử + chỉ báo kỹ thuật từ Close
    - OHLC của ngày t-1 được shift(1) trước khi dùng
    """
    df = df.copy()
    target = config.TARGET_COLUMN

    # Lag Close (t-1 → t-10)
    for lag in config.LAG_WINDOWS:
        df[f"Lag_{lag}"] = df[target].shift(lag)

    # Moving Averages từ Close
    for w in config.MA_WINDOWS:
        df[f"SMA_{w}"] = df[target].rolling(w).mean()
        df[f"EMA_{w}"] = df[target].ewm(span=w, adjust=False).mean()

    # RSI(14)
    delta = df[target].diff()
    gain  = delta.clip(lower=0).rolling(config.RSI_PERIOD).mean()
    loss  = (-delta.clip(upper=0)).rolling(config.RSI_PERIOD).mean()
    df["RSI"] = 100 - (100 / (1 + gain / (loss + 1e-9)))

    # Bollinger Bands
    mid = df[target].rolling(config.BB_PERIOD).mean()
    std = df[target].rolling(config.BB_PERIOD).std()
    df["BB_Upper"] = mid + config.BB_STD * std
    df["BB_Lower"] = mid - config.BB_STD * std
    df["BB_Width"] = df["BB_Upper"] - df["BB_Lower"]
    df["BB_Pct"]   = (df[target] - df["BB_Lower"]) / (df["BB_Width"] + 1e-9)

    # Returns & Volatility
    df["Return_1d"]  = df[target].pct_change(1)
    df["Return_5d"]  = df[target].pct_change(5)
    df["Volatility"] = df["Return_1d"].rolling(10).std()

    # High-Low range và candle body của ngày TRƯỚC (shift 1 — không leak)
    if all(c in df.columns for c in ["Open", "High", "Low"]):
        df["HL_Ratio_prev"] = (df["High"].shift(1) - df["Low"].shift(1)) / (df[target].shift(1) + 1e-9)
        df["OC_Ratio_prev"] = (df[target].shift(1) - df["Open"].shift(1)) / (df["Open"].shift(1) + 1e-9)

    # Volume của ngày trước
    if "Volume" in df.columns:
        df["Vol_MA5"]   = df["Volume"].rolling(5).mean()
        df["Vol_Ratio"] = df["Volume"].shift(1) / (df["Vol_MA5"].shift(1) + 1e-9)

    # Bỏ cột OHLCV gốc — không cho model thấy Open/High/Low hiện tại
    drop_cols = [c for c in ["Open", "High", "Low", "Volume", "ticker"] if c in df.columns]
    df = df.drop(columns=drop_cols)

    before = len(df)
    df = df.dropna().reset_index(drop=True)
    print(f"  Feature engineering: {df.shape[1]} cột, {len(df)} dòng "
          f"(loại {before - len(df)} dòng NaN)")
    return df


def split_data(df: pd.DataFrame):
    n       = len(df)
    n_train = int(n * config.TRAIN_RATIO)
    n_val   = int(n * config.VAL_RATIO)
    train   = df.iloc[:n_train].copy()
    val     = df.iloc[n_train:n_train + n_val].copy()
    test    = df.iloc[n_train + n_val:].copy()
    print(f"  Split → Train:{len(train)}  Val:{len(val)}  Test:{len(test)}")
    return train, val, test


def get_feature_columns(df: pd.DataFrame) -> list:
    exclude = {config.DATE_COLUMN, "ticker"}
    return [c for c in df.columns if c not in exclude]


def scale_data(train, val, test, features):
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
    Xs, ys = [], []
    for i in range(len(X) - seq_len):
        Xs.append(X[i:i + seq_len])
        ys.append(y[i + seq_len])
    return np.array(Xs, dtype=np.float32), np.array(ys, dtype=np.float32)
