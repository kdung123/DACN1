"""
app.py — Flask Training Dashboard
Tích hợp 6 cải tiến từ models_improved.py
"""

import os, sys, json, traceback, threading, queue
from flask import Flask, Response, request, jsonify
import numpy as np
import pandas as pd

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
sys.path.insert(0, os.path.dirname(__file__))

import config
import data_processing as dp
import evaluation as ev
import models as mi

app = Flask(__name__, static_folder="ui", static_url_path="/ui")

MODEL_REGISTRY = {
    # Tên hiển thị  →  id nội bộ
    "AutoARIMA":     "auto_arima",
    "Ridge(Optuna)": "optuna_ridge",
    "RF(Optuna)":    "optuna_rf",
    "LSTM+Attn":     "lstm_attention",
    "BiGRU":         "bigru",
    "Ensemble":      "ensemble",
}


@app.route("/")
def index():
    with open(os.path.join(os.path.dirname(__file__), "ui", "index.html"), encoding="utf-8") as f:
        return f.read()


@app.route("/api/tickers")
def api_tickers():
    result = []
    for t in config.VN30_TICKERS:
        try:
            df   = dp.preprocess(dp.load_stock_data(t, "historical"), t)
            last = df.iloc[-1]; prev = df.iloc[-2]
            chg  = float(last["Close"] - prev["Close"])
            pct  = chg / float(prev["Close"]) * 100
            result.append({"ticker": t, "close": float(last["Close"]),
                            "change": round(chg, 0), "pct": round(pct, 2),
                            "date": str(last["Date"])[:10]})
        except Exception:
            result.append({"ticker": t, "close": 0, "change": 0, "pct": 0, "date": ""})
    return jsonify(result)


@app.route("/api/chart/<ticker>")
def api_chart(ticker):
    n = int(request.args.get("n", 250))
    try:
        df = dp.preprocess(dp.load_stock_data(ticker, "historical"), ticker)
        if n > 0:
            df = df.tail(n)
        rows = df[["Date", "Close", "Volume"]].copy()
        rows["Date"]   = rows["Date"].astype(str).str[:10]
        rows["Close"]  = rows["Close"].round(0).astype(int)
        rows["Volume"] = rows["Volume"].astype(int)
        return jsonify(rows.to_dict("records"))
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/train", methods=["POST"])
def api_train():
    body    = request.json or {}
    ticker  = body.get("ticker", "VCB")
    models  = body.get("models", list(MODEL_REGISTRY.keys()))
    seq_len = int(body.get("seq_len", config.SEQUENCE_LEN))
    epochs  = int(body.get("epochs",  config.DL_EPOCHS))
    n_trials= int(body.get("n_trials", 20))

    q = queue.Queue()

    def send(t, **kw):
        q.put(json.dumps({"type": t, **kw}))

    def log(msg):
        send("log", msg=msg)

    def progress(model_id, pct):
        send("progress", model=model_id, pct=pct)

    def run():
        try:
            # ── Load & preprocess ──────────────────────────
            log(f"[1/5] Load dữ liệu {ticker}...")
            df = dp.preprocess(dp.load_stock_data(ticker, "historical"), ticker)
            log(f"      ✓ {len(df)} phiên ({str(df['Date'].min())[:10]} → {str(df['Date'].max())[:10]})")

            # ── Feature engineering ────────────────────────
            log("[2/5] Feature engineering...")
            df = dp.feature_engineering(df)
            log(f"      ✓ {df.shape[1]} đặc trưng, {len(df)} dòng")

            # ── Split ──────────────────────────────────────
            log("[3/5] Chia Train/Val/Test (70/15/15)...")
            train, val, test = dp.split_data(df)
            log(f"      ✓ Train={len(train)}  Val={len(val)}  Test={len(test)}")

            # ── Scale ──────────────────────────────────────
            log("[4/5] Chuẩn hóa...")
            features = dp.get_feature_columns(df)
            target   = config.TARGET_COLUMN
            (X_train, y_train, X_val, y_val, X_test, y_test,
             scaler_X, scaler_y) = dp.scale_data(train, val, test, features)

            y_test_real = test[target].values
            test_dates  = test["Date"].astype(str).str[:10].tolist()
            base_price  = float(val[target].iloc[-1])   # điểm khởi đầu cho return→price

            send("actual", dates=test_dates, values=[round(float(v)) for v in y_test_real])
            log("      ✓ Xong")

            log("[5/5] Huấn luyện các mô hình cải tiến...")
            results     = {}   # model_id → metrics dict
            predictions = {}   # model_id → {dates, values}

            # DL sequences
            dl_ready = False
            if any(m in models for m in ["LSTM+Attn", "BiGRU", "Ensemble"]):
                Xs_tr, ys_tr = dp.make_sequences(X_train, y_train, seq_len)
                Xs_vl, ys_vl = dp.make_sequences(X_val,   y_val,   seq_len)
                Xs_te, _     = dp.make_sequences(X_test,  y_test,  seq_len)
                y_te_seq     = y_test_real[seq_len:]
                td_seq       = test_dates[seq_len:]
                dl_ready     = len(Xs_tr) >= 10

            # ── 1. AUTO_ARIMA ──────────────────────────────
            if "AutoARIMA" in models:
                mid = "AutoARIMA"
                log(f"\n  → {mid}: auto_arima tìm (p,d,q)...")
                progress(mid, 5)
                try:
                    train_val_series = pd.concat([train, val])[target]
                    _, order = mi.train_auto_arima(
                        train_val_series,
                        send_log=lambda m: log(m)
                    )
                    progress(mid, 15)
                    pred = mi.predict_arima_walkforward(
                        train_val_series, y_test_real, order,
                        send_log=lambda m: log(m),
                        send_progress=lambda p: progress(mid, p)
                    )
                    r = _make_result(mid, y_test_real, pred, test_dates)
                    results[mid] = r; predictions[mid] = {"dates": test_dates, "values": pred.tolist()}
                    progress(mid, 100)
                    log(f"    ✓ {mid}  ARIMA{order}  " + _fmt_metrics(r))
                except Exception as e:
                    log(f"    ⚠ {mid} lỗi: {e}")

            # ── 2. RIDGE (OPTUNA) ──────────────────────────
            if "Ridge(Optuna)" in models:
                mid = "Ridge(Optuna)"
                log(f"\n  → {mid}: Optuna tìm alpha tốt nhất ({n_trials} trials)...")
                progress(mid, 5)
                try:
                    model, best_alpha = mi.train_ridge_optuna(
                        X_train, y_train, X_val, y_val,
                        n_trials=n_trials,
                        send_log=lambda m: log(m)
                    )
                    pred = scaler_y.inverse_transform(
                        model.predict(X_test).reshape(-1, 1)).ravel()
                    r = _make_result(mid, y_test_real, pred, test_dates)
                    results[mid] = r; predictions[mid] = {"dates": test_dates, "values": [round(float(v)) for v in pred]}
                    progress(mid, 100)
                    log(f"    ✓ {mid}  alpha={best_alpha:.4f}  " + _fmt_metrics(r))
                except Exception as e:
                    log(f"    ⚠ {mid} lỗi: {e}")

            # ── 3. RANDOM FOREST (OPTUNA) ──────────────────
            if "RF(Optuna)" in models:
                mid = "RF(Optuna)"
                log(f"\n  → {mid}: Optuna tìm hyperparameters ({n_trials} trials)...")
                progress(mid, 5)
                try:
                    model, bp = mi.train_rf_optuna(
                        X_train, y_train, X_val, y_val,
                        n_trials=n_trials,
                        send_log=lambda m: log(m)
                    )
                    pred = scaler_y.inverse_transform(
                        model.predict(X_test).reshape(-1, 1)).ravel()
                    r = _make_result(mid, y_test_real, pred, test_dates)
                    results[mid] = r; predictions[mid] = {"dates": test_dates, "values": [round(float(v)) for v in pred]}
                    progress(mid, 100)
                    log(f"    ✓ {mid}  {bp}  " + _fmt_metrics(r))
                except Exception as e:
                    log(f"    ⚠ {mid} lỗi: {e}")

            # ── 4. LSTM + ATTENTION ────────────────────────
            if "LSTM+Attn" in models and dl_ready:
                mid = "LSTM+Attn"
                log(f"\n  → {mid}: LSTM với custom Attention layer...")
                progress(mid, 3)
                try:
                    model = mi.train_lstm_attention(
                        Xs_tr, ys_tr, Xs_vl, ys_vl,
                        epochs=epochs,
                        send_log=lambda m: log(m),
                        send_progress=lambda p: progress(mid, p)
                    )
                    pred = scaler_y.inverse_transform(
                        model.predict(Xs_te, verbose=0)).ravel()
                    r = _make_result(mid, y_te_seq, pred, td_seq)
                    results[mid] = r; predictions[mid] = {"dates": td_seq, "values": [round(float(v)) for v in pred]}
                    progress(mid, 100)
                    log(f"    ✓ {mid}  " + _fmt_metrics(r))
                except Exception as e:
                    log(f"    ⚠ {mid} lỗi: {e}\n{traceback.format_exc()}")

            # ── 5. BIDIRECTIONAL GRU ───────────────────────
            if "BiGRU" in models and dl_ready:
                mid = "BiGRU"
                log(f"\n  → {mid}: Bidirectional GRU...")
                progress(mid, 3)
                try:
                    model = mi.train_bigru(
                        Xs_tr, ys_tr, Xs_vl, ys_vl,
                        epochs=epochs,
                        send_log=lambda m: log(m),
                        send_progress=lambda p: progress(mid, p)
                    )
                    pred = scaler_y.inverse_transform(
                        model.predict(Xs_te, verbose=0)).ravel()
                    r = _make_result(mid, y_te_seq, pred, td_seq)
                    results[mid] = r; predictions[mid] = {"dates": td_seq, "values": [round(float(v)) for v in pred]}
                    progress(mid, 100)
                    log(f"    ✓ {mid}  " + _fmt_metrics(r))
                except Exception as e:
                    log(f"    ⚠ {mid} lỗi: {e}")

            # ── 6. ENSEMBLE ────────────────────────────────
            if "Ensemble" in models and len(predictions) >= 2:
                mid = "Ensemble"
                log(f"\n  → {mid}: tìm trọng số tối ưu (Optuna)...")
                progress(mid, 5)
                try:
                    # Chỉ dùng các model đã chạy thành công
                    preds_for_ens = {k: v["values"] for k, v in predictions.items()}

                    # Tìm y_true tương ứng (lấy min độ dài)
                    min_len   = min(len(v) for v in preds_for_ens.values())
                    y_ens_val = y_test_real[-min_len:]

                    weights = mi.find_optimal_ensemble_weights(
                        preds_for_ens, y_ens_val,
                        send_log=lambda m: log(m)
                    )
                    progress(mid, 70)
                    ens_pred = mi.ensemble_predict(preds_for_ens, weights)
                    ens_dates = test_dates[-len(ens_pred):]

                    r = _make_result(mid, y_ens_val, ens_pred.astype(float), ens_dates)
                    results[mid] = r
                    predictions[mid] = {"dates": ens_dates, "values": ens_pred.tolist()}
                    progress(mid, 100)
                    log(f"    ✓ {mid}  weights={weights}  " + _fmt_metrics(r))
                except Exception as e:
                    log(f"    ⚠ {mid} lỗi: {e}")

            # ── Tổng kết ───────────────────────────────────
            results_list = list(results.values())
            send("done", results=results_list, predictions=predictions)
            log("━" * 42)
            log("✅ Huấn luyện hoàn tất!")
            if results_list:
                best = min(results_list, key=lambda r: r["rmse"])
                log(f"   Mô hình tốt nhất: {best['model']}  RMSE={best['rmse']:.0f}  R²={best['r2']:.4f}")

        except Exception:
            log("❌ Lỗi:\n" + traceback.format_exc())
        finally:
            q.put(None)

    threading.Thread(target=run, daemon=True).start()

    def stream():
        while True:
            item = q.get()
            if item is None:
                break
            yield f"data: {item}\n\n"

    return Response(stream(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# ── Helpers ──────────────────────────────────────────────────

def _make_result(model_id, y_true, y_pred, dates):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    n = min(len(y_true), len(y_pred))
    y_true, y_pred = y_true[-n:], y_pred[-n:]
    return {
        "model": model_id,
        "mae":   round(ev.mae(y_true, y_pred),  2),
        "rmse":  round(ev.rmse(y_true, y_pred), 2),
        "mape":  round(ev.mape(y_true, y_pred), 4),
        "r2":    round(ev.r2(y_true, y_pred),   4),
    }


def _fmt_metrics(r):
    return (f"MAE={r['mae']:.0f}  RMSE={r['rmse']:.0f}  "
            f"MAPE={r['mape']:.2f}%  R²={r['r2']:.4f}")


# ════════════════════════════════════════════════════════════
#  CRAWL — Server-Sent Events (streaming log)
# ════════════════════════════════════════════════════════════
@app.route("/api/crawl", methods=["POST"])
def api_crawl():
    """
    Crawl dữ liệu từ yfinance (Yahoo Finance) cho các mã VN30.
    Stream log realtime về UI.
    body: { tickers: [...], years: 5, source: "yfinance" }
    """
    body    = request.json or {}
    tickers = body.get("tickers", config.VN30_TICKERS)
    years   = int(body.get("years", 5))
    source  = body.get("source", "yfinance")   # "yfinance" | "vnstock"

    q = queue.Queue()

    def send(t, **kw):
        q.put(json.dumps({"type": t, **kw}))

    def log(msg, cls=""):
        send("log", msg=msg, cls=cls)

    def run():
        try:
            import yfinance as yf
            from datetime import datetime, timedelta
            import pathlib

            end_date   = datetime.now()
            start_date = end_date - timedelta(days=365 * years)
            start_str  = start_date.strftime("%Y-%m-%d")
            end_str    = end_date.strftime("%Y-%m-%d")

            base_dir = pathlib.Path(config.DATA_DIR)

            log(f"📡 Bắt đầu crawl {len(tickers)} mã VN30")
            log(f"   Nguồn: Yahoo Finance ({source})")
            log(f"   Khoảng thời gian: {start_str} → {end_str} ({years} năm)")
            log(f"   Lưu vào: {base_dir.resolve()}")
            log("─" * 44)

            # Tạo thư mục
            for t in tickers:
                (base_dir / t).mkdir(parents=True, exist_ok=True)

            success, failed = [], []

            for i, ticker in enumerate(tickers):
                try:
                    send("crawl_progress", index=i, total=len(tickers),
                         ticker=ticker, status="downloading")
                    log(f"[{i+1:02d}/{len(tickers)}] {ticker} đang tải...")

                    # Download từ Yahoo Finance
                    ticker_yf = f"{ticker}.VN"
                    df = yf.download(
                        ticker_yf,
                        start=start_str,
                        end=end_str,
                        progress=False,
                        auto_adjust=True,
                    )

                    if df.empty:
                        raise ValueError("Không có dữ liệu trả về")

                    # Chuẩn hóa cột — yfinance trả về MultiIndex hoặc flat
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = df.columns.get_level_values(0)
                    df = df.reset_index()
                    df.columns = [c.strip() for c in df.columns]

                    # Đổi tên để khớp với data_processing.py
                    col_map = {
                        "Datetime": "Date", "datetime": "Date",
                        "Open":  "Open",  "High": "High",
                        "Low":   "Low",   "Close": "Close",
                        "Volume":"Volume",
                    }
                    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
                    df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")

                    # Chỉ giữ cột cần thiết
                    keep = [c for c in ["Date","Open","High","Low","Close","Volume"] if c in df.columns]
                    df   = df[keep].dropna(subset=["Close"])

                    # Lưu file historical (toàn bộ) + Ny
                    hist_path = base_dir / ticker / f"{ticker}_historical.csv"
                    year_path = base_dir / ticker / f"{ticker}_{years}y.csv"
                    df.to_csv(hist_path, index=False)
                    df.to_csv(year_path, index=False)

                    rows = len(df)
                    last = df["Date"].iloc[-1] if rows else "?"
                    success.append(ticker)
                    send("crawl_progress", index=i+1, total=len(tickers),
                         ticker=ticker, status="done", rows=rows)
                    log(f"         ✓ {rows} phiên  (đến {last})")

                except Exception as e:
                    failed.append(ticker)
                    send("crawl_progress", index=i+1, total=len(tickers),
                         ticker=ticker, status="error")
                    log(f"         ⚠ Lỗi: {e}")

            log("─" * 44)
            log(f"✅ Hoàn tất!  Thành công: {len(success)}/{len(tickers)}")
            if failed:
                log(f"   ❌ Thất bại: {', '.join(failed)}")

            send("crawl_done", success=success, failed=failed)

        except ImportError:
            log("❌ Chưa cài yfinance — chạy: pip install yfinance")
            send("crawl_done", success=[], failed=tickers)
        except Exception:
            log("❌ Lỗi:\n" + traceback.format_exc())
            send("crawl_done", success=[], failed=tickers)
        finally:
            q.put(None)

    threading.Thread(target=run, daemon=True).start()

    def stream():
        while True:
            item = q.get()
            if item is None:
                break
            yield f"data: {item}\n\n"

    return Response(stream(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


if __name__ == "__main__":
    print("\n🚀  VN30 Training Dashboard — Improved Models")
    print("    http://localhost:5000\n")
    app.run(debug=False, threaded=True, port=5000)
