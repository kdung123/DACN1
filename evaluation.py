# ╔══════════════════════════════════════════════════════════════╗
# ║                      evaluation.py                          ║
# ║            Metrics · Biểu đồ · Export kết quả              ║
# ╚══════════════════════════════════════════════════════════════╝

import os

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

matplotlib.use("Agg")

import config

# ── Thiết lập style chung ────────────────────────────────────────
plt.rcParams.update({
    "font.family":    "DejaVu Sans",
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

# Màu sắc cho từng model (khớp với index.html)
MODEL_COLORS = {
    "Ridge(Optuna)": "#2563eb",
    "RF(Optuna)":    "#16a34a",
    "BiGRU":         "#db2777",
    "Ensemble":      "#0891b2",
}
PALETTE = list(MODEL_COLORS.values())


# ════════════════════════════════════════════════════════════════
#  1. METRICS
# ════════════════════════════════════════════════════════════════

def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error — đơn vị VNĐ, dễ diễn giải."""
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error — phạt nặng sai số lớn (spike)."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Percentage Error — % sai số trung bình.
    Bỏ qua các điểm y_true = 0 để tránh chia 0."""
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """R² (hệ số xác định) — 1.0 = hoàn hảo, < 0 = tệ hơn đường trung bình."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1 - ss_res / (ss_tot + 1e-10))


def evaluate(y_true: np.ndarray, y_pred: np.ndarray, name: str) -> dict:
    """
    Tính toán và in 4 chỉ số cho một model.

    Returns
    -------
    dict  {"Model", "MAE", "RMSE", "MAPE", "R2"}
    """
    result = {
        "Model": name,
        "MAE":   round(mae(y_true, y_pred),  2),
        "RMSE":  round(rmse(y_true, y_pred), 2),
        "MAPE":  round(mape(y_true, y_pred), 4),
        "R2":    round(r2(y_true, y_pred),   4),
    }
    print(f"    {name:<22}  "
          f"MAE={result['MAE']:>8.0f}  "
          f"RMSE={result['RMSE']:>8.0f}  "
          f"MAPE={result['MAPE']:.2f}%  "
          f"R²={result['R2']:.4f}")
    return result


def compare_models(results: list) -> pd.DataFrame:
    """
    In bảng so sánh tất cả model, sắp xếp theo RMSE tăng dần.

    Returns
    -------
    pd.DataFrame   index = Model name
    """
    df  = pd.DataFrame(results).set_index("Model").sort_values("RMSE")
    sep = "─" * 72
    print(f"\n{sep}")
    print("  SO SÁNH MÔ HÌNH  (↑ RMSE thấp hơn = tốt hơn)")
    print(sep)
    print(df.to_string())
    print(sep + "\n")
    return df


def save_results_csv(df_metrics: pd.DataFrame, ticker: str):
    """Lưu bảng metrics ra file CSV."""
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    path = os.path.join(config.RESULTS_DIR, f"metrics_{ticker}.csv")
    df_metrics.to_csv(path)
    print(f"  ✓ Lưu metrics → {path}")


# ════════════════════════════════════════════════════════════════
#  2. BIỂU ĐỒ
# ════════════════════════════════════════════════════════════════

def _savefig(fig, fname: str) -> str:
    """Lưu figure vào thư mục results/ và đóng."""
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    path = os.path.join(config.RESULTS_DIR, fname)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_close_history(df: pd.DataFrame, ticker: str) -> str:
    """
    Vẽ lịch sử giá đóng cửa và khối lượng giao dịch.
    Dùng cho EDA trước khi train.
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)

    # Giá Close
    axes[0].plot(df[config.DATE_COLUMN], df[config.TARGET_COLUMN],
                 color="#2563eb", lw=1.2)
    axes[0].set_ylabel("Giá đóng cửa (VNĐ)")
    axes[0].set_title(f"{ticker} — Lịch sử giá đóng cửa")
    axes[0].yaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda x, _: f"{x:,.0f}")
    )

    # Khối lượng
    if "Volume" in df.columns:
        axes[1].bar(df[config.DATE_COLUMN], df["Volume"],
                    color="#93c5fd", width=1, alpha=0.8)
        axes[1].set_ylabel("Khối lượng")

    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.xticks(rotation=30)
    fig.tight_layout()
    return _savefig(fig, f"eda_{ticker}.png")


def plot_predictions(dates, y_true, predictions: dict, ticker: str) -> str:
    """
    Vẽ đường dự báo của tất cả model chồng lên đường thực tế.

    Lưu ý alignment:
      BiGRU bắt đầu muộn hơn seq_len phiên so với các model khác
      → đường thực tế (đen) luôn vẽ đủ toàn bộ test period
      → pred được align về cuối timeline bằng null padding

    Parameters
    ----------
    dates       : array-like   Mảng ngày tháng của test set
    y_true      : np.ndarray   Giá thực tế
    predictions : dict         {model_name: np.ndarray}
    ticker      : str
    """
    fig, ax = plt.subplots(figsize=(14, 5))

    dates   = np.array(dates)
    y_true  = np.array(y_true)
    n_dates = len(dates)

    # ── Đường thực tế (toàn bộ test period) ──────────────────
    ax.plot(dates, y_true,
            label="Thực tế", color="black", lw=2.0, zorder=5)

    # ── Đường dự báo từng model ───────────────────────────────
    for (name, pred), color in zip(predictions.items(),
                                   [MODEL_COLORS.get(n, c)
                                    for n, c in zip(predictions, PALETTE)]):
        pred = np.array(pred)
        n    = min(n_dates, len(pred))

        ax.plot(dates[-n:], pred[-n:],
                label=f"{name}",
                color=color, lw=1.4, linestyle="--", alpha=0.9)

        # Đánh dấu điểm bắt đầu dự báo (BiGRU bắt đầu muộn hơn)
        if n < n_dates:
            ax.axvline(x=dates[-n], color=color,
                       lw=0.8, linestyle=":", alpha=0.4)

    ax.set_title(f"Dự báo giá đóng cửa — {ticker}  (Tập Test 15%)")
    ax.set_xlabel("Ngày")
    ax.set_ylabel("Giá (VNĐ)")
    ax.yaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda x, _: f"{x:,.0f}")
    )
    ax.legend(loc="upper left", fontsize=9, framealpha=0.8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.xticks(rotation=30)
    fig.tight_layout()
    return _savefig(fig, f"predictions_{ticker}.png")


def plot_training_history(histories: dict, ticker: str) -> str | None:
    """
    Vẽ loss curve (train vs val) cho các DL model.

    Hiển thị:
      • Đường xanh  = train loss
      • Đường đỏ    = val loss
      • ★ xanh lá   = epoch val_loss tốt nhất (restore_best_weights)
      • ▲ cam       = epoch EarlyStopping dừng (nếu khác best)

    Parameters
    ----------
    histories : dict   {model_name: keras.callbacks.History}
    """
    if not histories:
        return None

    n_models = len(histories)
    fig, axes = plt.subplots(1, n_models,
                              figsize=(6 * n_models, 4),
                              squeeze=False)

    for ax, (name, hist) in zip(axes[0], histories.items()):
        train_loss = hist.history.get("loss", [])
        val_loss   = hist.history.get("val_loss", [])
        epochs     = list(range(1, len(train_loss) + 1))

        ax.plot(epochs, train_loss, label="Train loss",
                color="#2563eb", lw=1.6)
        ax.plot(epochs, val_loss,   label="Val loss",
                color="#dc2626", lw=1.6, linestyle="--")

        # ★ Best val epoch
        best_ep = int(np.argmin(val_loss)) + 1
        best_vl = val_loss[best_ep - 1]
        ax.scatter([best_ep], [best_vl],
                   color="#16a34a", s=90, zorder=6,
                   label=f"Best val (ep {best_ep})")
        ax.axvline(x=best_ep, color="#16a34a",
                   lw=0.9, linestyle="--", alpha=0.5)

        # ▲ EarlyStopping epoch (nếu khác best)
        stopped_ep = len(train_loss)
        if stopped_ep != best_ep:
            ax.axvline(x=stopped_ep, color="#f59e0b",
                       lw=0.9, linestyle=":", alpha=0.7,
                       label=f"Stopped (ep {stopped_ep})")

        ax.set_title(f"{name} — Loss curve")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE Loss")
        ax.legend(fontsize=8)
        ax.set_xlim(1, max(epochs) + 0.5)

    fig.suptitle(f"Training History — {ticker}", fontsize=13)
    fig.tight_layout()
    return _savefig(fig, f"training_history_{ticker}.png")


def plot_metrics_comparison(df_metrics: pd.DataFrame, ticker: str) -> str:
    """
    Vẽ bar chart so sánh MAE / RMSE / MAPE / R² của tất cả model.
    """
    metrics = [m for m in ["MAE", "RMSE", "MAPE", "R2"]
               if m in df_metrics.columns]
    fig, axes = plt.subplots(1, len(metrics),
                              figsize=(4 * len(metrics), 4))
    if len(metrics) == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        vals   = df_metrics[metric]
        colors = [MODEL_COLORS.get(idx, "#94a3b8") for idx in vals.index]
        bars   = ax.bar(vals.index, vals.values,
                        color=colors, edgecolor="white", linewidth=0.8)

        ax.set_title(metric, fontweight="bold")
        ax.set_xticklabels(vals.index, rotation=25, ha="right", fontsize=9)

        # Label giá trị trên mỗi cột
        for bar, v in zip(bars, vals.values):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() * 1.01,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=8)

    fig.suptitle(f"So sánh chỉ số đánh giá — {ticker}", fontsize=13)
    fig.tight_layout()
    return _savefig(fig, f"metrics_{ticker}.png")
