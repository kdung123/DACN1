# ============================================================
# evaluation.py  –  Metrics, plots, export
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import config

plt.rcParams.update({"font.family": "DejaVu Sans", "axes.titlesize": 12, "axes.labelsize": 10})

PALETTE = ["#2196F3", "#F44336", "#4CAF50", "#9C27B0", "#FF9800", "#00BCD4"]

# ─────────────────────────────────────────────
# Chỉ số
# ─────────────────────────────────────────────

def mae(y, p):
    return float(np.mean(np.abs(y - p)))

def rmse(y, p):
    return float(np.sqrt(np.mean((y - p) ** 2)))

def mape(y, p):
    mask = y != 0
    return float(np.mean(np.abs((y[mask] - p[mask]) / y[mask])) * 100)

def r2(y, p):
    ss_res = np.sum((y - p) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return float(1 - ss_res / (ss_tot + 1e-10))

def evaluate(y_true: np.ndarray, y_pred: np.ndarray, name: str) -> dict:
    r = {
        "Model": name,
        "MAE":  round(mae(y_true, y_pred),  2),
        "RMSE": round(rmse(y_true, y_pred), 2),
        "MAPE": round(mape(y_true, y_pred), 4),
        "R2":   round(r2(y_true, y_pred),   4),
    }
    print(f"    {name:<22} MAE={r['MAE']:>8.0f}  RMSE={r['RMSE']:>8.0f}  "
          f"MAPE={r['MAPE']:.2f}%  R²={r['R2']:.4f}")
    return r


def compare_models(results: list) -> pd.DataFrame:
    df = pd.DataFrame(results).set_index("Model").sort_values("RMSE")
    sep = "─" * 68
    print(f"\n{sep}")
    print("  SO SÁNH MÔ HÌNH  (sắp xếp theo RMSE tăng dần)")
    print(sep)
    print(df.to_string())
    print(sep + "\n")
    return df


# ─────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────

def _savefig(fig, fname: str):
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    path = os.path.join(config.RESULTS_DIR, fname)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_predictions(dates, y_true, predictions: dict, ticker: str):
    """
    Vẽ biểu đồ dự báo vs thực tế.

    Fix: các model DL có pred ngắn hơn do SEQUENCE_LEN → align từ cuối
    nhưng vẫn vẽ TOÀN BỘ đường thực tế (màu đen) để người dùng thấy
    full test period. Phần pred được overlay đúng vị trí thời gian.
    """
    fig, ax = plt.subplots(figsize=(14, 5))

    dates   = np.array(dates)
    y_true  = np.array(y_true)
    n_dates = len(dates)

    # Vẽ toàn bộ đường thực tế
    ax.plot(dates, y_true, label="Thực tế", color="black", lw=1.8, zorder=5)

    for (name, pred), color in zip(predictions.items(), PALETTE):
        pred = np.array(pred)
        n    = min(n_dates, len(pred))

        # Align cuối: pred[-n:] khớp với dates[-n:] và y_true[-n:]
        ax.plot(
            dates[-n:], pred[-n:],
            label=f"{name} (n={n})",
            color=color, lw=1.3, linestyle="--", alpha=0.85,
        )

        # Đánh dấu điểm bắt đầu dự báo nếu pred ngắn hơn test
        if n < n_dates:
            ax.axvline(x=dates[-n], color=color, lw=0.7,
                       linestyle=":", alpha=0.5)

    ax.set_title(f"Dự báo giá đóng cửa — {ticker} (Tập Test)")
    ax.set_xlabel("Ngày")
    ax.set_ylabel("Giá (VNĐ)")
    ax.legend(loc="upper left", fontsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.xticks(rotation=30)
    fig.tight_layout()
    return _savefig(fig, f"predictions_{ticker}.png")


def plot_training_history(histories: dict, ticker: str):
    """
    Vẽ loss curve của các DL model (LSTM+Attention, BiGRU).

    histories = {
        "LSTM+Attention": keras_history_object,
        "BiGRU":          keras_history_object,
    }

    Hiển thị:
    - train_loss và val_loss theo epoch
    - Đánh dấu epoch EarlyStopping dừng (đường đứt dọc)
    - Đánh dấu epoch val_loss tốt nhất (dấu ★)
    """
    if not histories:
        return None

    n_models = len(histories)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 4),
                              squeeze=False)

    for ax, (name, hist) in zip(axes[0], histories.items()):
        train_loss = hist.history.get("loss", [])
        val_loss   = hist.history.get("val_loss", [])
        epochs     = list(range(1, len(train_loss) + 1))

        ax.plot(epochs, train_loss, label="Train loss", color="#2196F3", lw=1.5)
        ax.plot(epochs, val_loss,   label="Val loss",   color="#F44336", lw=1.5)

        # Epoch val_loss tốt nhất
        best_ep = int(np.argmin(val_loss)) + 1
        best_vl = val_loss[best_ep - 1]
        ax.scatter([best_ep], [best_vl], color="#4CAF50", s=80,
                   zorder=6, label=f"Best val (ep {best_ep})")
        ax.axvline(x=best_ep, color="#4CAF50", lw=0.8, linestyle="--", alpha=0.6)

        # Epoch thực tế dừng (EarlyStopping có thể dừng sau best_ep)
        stopped_ep = len(train_loss)
        if stopped_ep != best_ep:
            ax.axvline(x=stopped_ep, color="#FF9800", lw=0.8,
                       linestyle=":", alpha=0.7, label=f"Stopped (ep {stopped_ep})")

        ax.set_title(f"{name} — Loss curve")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE Loss")
        ax.legend(fontsize=8)
        ax.set_xlim(1, max(epochs) + 0.5)

    fig.suptitle(f"Training History — {ticker}", fontsize=13)
    fig.tight_layout()
    return _savefig(fig, f"training_history_{ticker}.png")


def plot_metrics_comparison(df_metrics: pd.DataFrame, ticker: str):
    metrics = [m for m in ["MAE", "RMSE", "MAPE", "R2"] if m in df_metrics.columns]
    fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 4))
    if len(metrics) == 1:
        axes = [axes]
    for ax, metric in zip(axes, metrics):
        vals = df_metrics[metric]
        bars = ax.bar(vals.index, vals.values,
                      color=PALETTE[:len(vals)], edgecolor="white")
        ax.set_title(metric)
        ax.set_xticklabels(vals.index, rotation=25, ha="right", fontsize=9)
        for bar, v in zip(bars, vals.values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.01,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    fig.suptitle(f"So sánh chỉ số đánh giá — {ticker}", fontsize=13)
    fig.tight_layout()
    return _savefig(fig, f"metrics_{ticker}.png")


def plot_close_history(df: pd.DataFrame, ticker: str):
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    axes[0].plot(df[config.DATE_COLUMN], df[config.TARGET_COLUMN],
                 color="#2196F3", lw=1.2)
    axes[0].set_ylabel("Giá đóng cửa (VNĐ)")
    axes[0].set_title(f"{ticker} – Lịch sử giá đóng cửa")
    if "Volume" in df.columns:
        axes[1].bar(df[config.DATE_COLUMN], df["Volume"],
                    color="#90CAF9", width=1, alpha=0.7)
        axes[1].set_ylabel("Khối lượng")
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.xticks(rotation=30)
    fig.tight_layout()
    return _savefig(fig, f"eda_{ticker}.png")


def save_results_csv(df_metrics: pd.DataFrame, ticker: str):
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    path = os.path.join(config.RESULTS_DIR, f"metrics_{ticker}.csv")
    df_metrics.to_csv(path)
    print(f"  ✓ {path}")
