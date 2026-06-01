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
    """
    R² = 1 - SS_res / SS_tot
    = 1 — giải thích được 0% phương sai (tệ như dự báo mean)
    = 1 — giải thích được 100% phương sai (hoàn hảo)
    Có thể âm nếu mô hình tệ hơn cả dự báo mean.
    """
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
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(dates, y_true, label="Thực tế", color="black", lw=1.8, zorder=5)
    palette = ["#2196F3", "#F44336", "#4CAF50", "#9C27B0", "#FF9800"]
    for (name, pred), color in zip(predictions.items(), palette):
        n = min(len(dates), len(pred))
        ax.plot(dates[-n:], pred[-n:], label=name, color=color, lw=1.3, linestyle="--", alpha=0.85)
    ax.set_title(f"Dự báo giá đóng cửa — {ticker} (Tập Test)")
    ax.set_xlabel("Ngày"); ax.set_ylabel("Giá (VNĐ)")
    ax.legend(loc="upper left")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.xticks(rotation=30); fig.tight_layout()
    return _savefig(fig, f"predictions_{ticker}.png")


def plot_metrics_comparison(df_metrics: pd.DataFrame, ticker: str):
    metrics = ["MAE", "RMSE", "MAPE", "R2"]
    metrics = [m for m in metrics if m in df_metrics.columns]
    fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 4))
    palette = ["#2196F3", "#F44336", "#4CAF50", "#9C27B0", "#FF9800"]
    for ax, metric in zip(axes, metrics):
        vals = df_metrics[metric]
        bars = ax.bar(vals.index, vals.values, color=palette[:len(vals)], edgecolor="white")
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
    axes[0].plot(df[config.DATE_COLUMN], df[config.TARGET_COLUMN], color="#2196F3", lw=1.2)
    axes[0].set_ylabel("Giá đóng cửa (VNĐ)")
    axes[0].set_title(f"{ticker} – Lịch sử giá đóng cửa")
    if "Volume" in df.columns:
        axes[1].bar(df[config.DATE_COLUMN], df["Volume"], color="#90CAF9", width=1, alpha=0.7)
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
