# ╔══════════════════════════════════════════════════════════════╗
# ║                          app.py                             ║
# ║           Flask Web Server — VN30 Training Dashboard        ║
# ╠══════════════════════════════════════════════════════════════╣
# ║  Luồng hoạt động (từ trên xuống):                          ║
# ║                                                            ║
# ║  [KHỞI ĐỘNG]                                               ║
# ║    1. Import thư viện & modules                            ║
# ║    2. Khởi tạo Flask app                                   ║
# ║    3. Định nghĩa helper functions                          ║
# ║                                                            ║
# ║  [API ENDPOINTS]                                           ║
# ║    GET  /                      → Serve index.html          ║
# ║    GET  /api/tickers           → Danh sách VN30 + giá     ║
# ║    GET  /api/chart/<ticker>    → Lịch sử giá + SMA20      ║
# ║    POST /api/train             → Train models (SSE stream) ║
# ║    POST /api/crawl             → Crawl Yahoo Finance (SSE) ║
# ║                                                            ║
# ║  [PIPELINE TRAIN — chạy trong thread riêng]               ║
# ║    Step 1 → Load CSV từ data/VN30_Data/<ticker>/           ║
# ║    Step 2 → Feature Engineering (30 features, no leak)    ║
# ║    Step 3 → Split 70/15/15 theo thứ tự thời gian          ║
# ║    Step 4 → MinMaxScaler (fit CHỈ trên train)             ║
# ║    Step 5 → Train: Ridge → RF → BiGRU → Ensemble          ║
# ║    Step 6 → Stream kết quả về UI qua SSE                  ║
# ╚══════════════════════════════════════════════════════════════╝


# ════════════════════════════════════════════════════════════════
#  PHẦN 1 — IMPORT
# ════════════════════════════════════════════════════════════════

import os
import sys
import json
import queue
import threading
import traceback

import numpy as np
import pandas as pd
from flask import Flask, Response, jsonify, request

# Tắt log TensorFlow không cần thiết
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Đảm bảo Python tìm được các module trong cùng thư mục
sys.path.insert(0, os.path.dirname(__file__))

# Import các module nội bộ
import config                   # Cấu hình toàn cục
import data_processing as dp    # Load + Feature Engineering + Split + Scale
import evaluation as ev         # Metrics (MAE, RMSE, MAPE, R²)
import models as mi             # Ridge, RF, BiGRU, Ensemble


# ════════════════════════════════════════════════════════════════
#  PHẦN 2 — KHỞI TẠO FLASK
# ════════════════════════════════════════════════════════════════

# Static folder trỏ đến thư mục ui/ chứa index.html
app = Flask(__name__, static_folder="ui", static_url_path="/ui")

# Danh sách model hỗ trợ — tên hiển thị → id nội bộ
MODEL_REGISTRY = {
    "Ridge(Optuna)": "optuna_ridge",
    "RF(Optuna)":    "optuna_rf",
    "BiGRU":         "bigru",
    "Ensemble":      "ensemble",
}


# ════════════════════════════════════════════════════════════════
#  PHẦN 3 — HELPER FUNCTIONS
#  Dùng chung cho nhiều endpoint — định nghĩa trước khi dùng
# ════════════════════════════════════════════════════════════════

def _make_result(model_id: str, y_true, y_pred, dates) -> dict:
    """
    Tính 4 chỉ số đánh giá và đóng gói thành dict.
    Tự động căn chỉnh độ dài y_true / y_pred về min.

    Returns
    -------
    dict  {"model", "mae", "rmse", "mape", "r2"}
    """
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    n      = min(len(y_true), len(y_pred))
    y_true, y_pred = y_true[-n:], y_pred[-n:]
    return {
        "model": model_id,
        "mae":   round(ev.mae(y_true, y_pred),  2),
        "rmse":  round(ev.rmse(y_true, y_pred), 2),
        "mape":  round(ev.mape(y_true, y_pred), 4),
        "r2":    round(ev.r2(y_true, y_pred),   4),
    }


def _fmt_metrics(r: dict) -> str:
    """Format dict metrics thành chuỗi ngắn để log."""
    return (f"MAE={r['mae']:.0f}  RMSE={r['rmse']:.0f}  "
            f"MAPE={r['mape']:.2f}%  R²={r['r2']:.4f}")


def _stream_response(q: queue.Queue) -> Response:
    """
    Generator đọc từ Queue và yield SSE events.
    None trong queue = tín hiệu kết thúc stream.
    """
    def generate():
        while True:
            item = q.get()
            if item is None:
                break
            yield f"data: {item}\n\n"

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ════════════════════════════════════════════════════════════════
#  PHẦN 4 — ENDPOINT: TRANG CHỦ
# ════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    """Serve file ui/index.html khi truy cập http://localhost:5000"""
    path = os.path.join(os.path.dirname(__file__), "ui", "index.html")
    with open(path, encoding="utf-8") as f:
        return f.read()


# ════════════════════════════════════════════════════════════════
#  PHẦN 5 — ENDPOINT: DANH SÁCH VN30
# ════════════════════════════════════════════════════════════════

@app.route("/api/tickers")
def api_tickers():
    """
    Trả về danh sách 30 mã VN30 kèm giá đóng cửa mới nhất.

    Response: JSON array
      [{ ticker, close, change, pct, date }, ...]
    """
    result = []

    for ticker in config.VN30_TICKERS:
        try:
            # Load và làm sạch dữ liệu
            df   = dp.preprocess(dp.load_stock_data(ticker, "historical"), ticker)
            last = df.iloc[-1]
            prev = df.iloc[-2]

            # Tính thay đổi giá so với phiên trước
            chg = float(last["Close"] - prev["Close"])
            pct = chg / float(prev["Close"]) * 100

            result.append({
                "ticker": ticker,
                "close":  float(last["Close"]),
                "change": round(chg, 0),
                "pct":    round(pct, 2),
                "date":   str(last["Date"])[:10],
            })
        except Exception:
            # Nếu không load được, trả về giá trị rỗng
            result.append({"ticker": ticker, "close": 0,
                           "change": 0, "pct": 0, "date": ""})

    return jsonify(result)


# ════════════════════════════════════════════════════════════════
#  PHẦN 6 — ENDPOINT: LỊCH SỬ GIÁ (cho biểu đồ EDA)
# ════════════════════════════════════════════════════════════════

@app.route("/api/chart/<ticker>")
def api_chart(ticker: str):
    """
    Trả về lịch sử giá đóng cửa + SMA20 cho biểu đồ EDA.

    Query params:
      n (int) : Số phiên gần nhất cần lấy (mặc định 250, 0 = tất cả)

    Response: JSON array
      [{ Date, Close, Volume, SMA20 }, ...]
    """
    n = int(request.args.get("n", 250))

    try:
        # Load và làm sạch dữ liệu gốc (chưa feature engineering)
        df = dp.preprocess(dp.load_stock_data(ticker, "historical"), ticker)

        # Tính SMA20 trên toàn bộ trước khi cắt — tránh SMA bị NaN đầu
        df["SMA20"] = df["Close"].rolling(20).mean().round(0)

        # Cắt n phiên gần nhất nếu cần
        if n > 0:
            df = df.tail(n)

        # Chỉ giữ cột cần thiết và chuẩn hóa kiểu dữ liệu
        rows = df[["Date", "Close", "Volume", "SMA20"]].copy()
        rows["Date"]   = rows["Date"].astype(str).str[:10]
        rows["Close"]  = rows["Close"].round(0).astype(int)
        rows["Volume"] = rows["Volume"].fillna(0).astype(int)
        rows["SMA20"]  = rows["SMA20"].fillna(0).astype(int)

        return jsonify(rows.to_dict("records"))

    except Exception as e:
        return jsonify({"error": str(e)}), 400


# ════════════════════════════════════════════════════════════════
#  PHẦN 7 — ENDPOINT: TRAIN MODELS (Server-Sent Events)
# ════════════════════════════════════════════════════════════════

@app.route("/api/train", methods=["POST"])
def api_train():
    """
    Nhận yêu cầu train từ UI, chạy pipeline trong thread riêng,
    stream log + kết quả về UI theo thời gian thực qua SSE.

    Request body (JSON):
      ticker   : str   Mã cổ phiếu (vd: "VCB")
      models   : list  Danh sách model cần train
      seq_len  : int   Sequence length cho BiGRU (mặc định 30)
      epochs   : int   Số epochs BiGRU (mặc định 100)
      n_trials : int   Số trials Optuna (mặc định 20)

    SSE events:
      log              → {msg}                  Log text
      progress         → {model, pct}           Tiến độ 0–100
      actual           → {dates, values}        Giá thực tế test set
      training_history → {model, train_loss, val_loss, stopped_epoch}
      done             → {results, predictions} Kết quả cuối cùng
    """
    # ── Đọc tham số từ request ────────────────────────────────
    body     = request.json or {}
    ticker   = body.get("ticker",   "VCB")
    models   = body.get("models",   list(MODEL_REGISTRY.keys()))
    seq_len  = int(body.get("seq_len",  config.SEQUENCE_LEN))
    epochs   = int(body.get("epochs",   config.DL_EPOCHS))
    n_trials = int(body.get("n_trials", 20))
    years    = int(body.get("years",    0))   # 0 = dùng toàn bộ dữ liệu

    # ── Queue SSE — thread train ghi vào, stream đọc ra ──────
    q = queue.Queue()

    def send(event_type: str, **kwargs):
        """Đẩy SSE event vào queue."""
        q.put(json.dumps({"type": event_type, **kwargs}))

    def log(msg: str):
        """Log một dòng text về UI."""
        send("log", msg=msg)

    def progress(model_id: str, pct: int):
        """Cập nhật thanh tiến độ của model."""
        send("progress", model=model_id, pct=pct)

    # ── Pipeline train — chạy trong thread riêng ─────────────
    def run():
        try:

            # ────────────────────────────────────────────────
            #  STEP 1 — LOAD DỮ LIỆU
            #  Đọc CSV từ data/VN30_Data/<ticker>/<ticker>_historical.csv
            # ────────────────────────────────────────────────
            log(f"[1/5] Load dữ liệu {ticker}...")
            df_raw = dp.load_stock_data(ticker, "historical")
            df     = dp.preprocess(df_raw, ticker)

            # ── Lọc theo số năm được chọn ─────────────────
            # years=0 → dùng toàn bộ lịch sử
            # years=N → chỉ lấy N năm gần nhất (~N×250 phiên)
            if years > 0:
                cutoff = pd.Timestamp.now() - pd.DateOffset(years=years)
                df = df[df["Date"] >= cutoff].reset_index(drop=True)
                log(f"      ✓ Lọc {years} năm gần nhất: {len(df)} phiên  "
                    f"({str(df['Date'].min())[:10]} → {str(df['Date'].max())[:10]})")
            else:
                log(f"      ✓ Toàn bộ lịch sử: {len(df)} phiên  "
                    f"({str(df['Date'].min())[:10]} → {str(df['Date'].max())[:10]})")

            if len(df) < 100:
                log(f"      ❌ Quá ít dữ liệu ({len(df)} phiên). Cần ít nhất 100 phiên.")
                q.put(None)
                return

            # ────────────────────────────────────────────────
            #  STEP 2 — FEATURE ENGINEERING
            #  Tạo 30 features kỹ thuật — tất cả từ Close.shift(1)
            #  để tránh lagged variable bias & data leakage
            # ────────────────────────────────────────────────
            log("[2/5] Feature engineering...")
            df = dp.feature_engineering(df)
            log(f"      ✓ {df.shape[1]} đặc trưng, {len(df)} dòng")

            # ────────────────────────────────────────────────
            #  STEP 3 — SPLIT 70 / 15 / 15
            #  Chia theo thứ tự thời gian — KHÔNG shuffle
            #  Train: ngày cũ nhất → Val → Test: ngày mới nhất
            # ────────────────────────────────────────────────
            log("[3/5] Chia Train / Val / Test  (70 / 15 / 15)...")
            train, val, test = dp.split_data(df)
            log(f"      ✓ Train={len(train)}  Val={len(val)}  Test={len(test)}")

            # ────────────────────────────────────────────────
            #  STEP 4 — CHUẨN HÓA MinMaxScaler
            #  scaler.fit() CHỈ trên train — val/test chỉ transform
            #  → tránh data leakage từ phân phối val/test vào scaler
            # ────────────────────────────────────────────────
            log("[4/5] Chuẩn hóa MinMaxScaler...")
            features = dp.get_feature_columns(df)
            target   = config.TARGET_COLUMN

            (X_train, y_train,
             X_val,   y_val,
             X_test,  y_test,
             scaler_X, scaler_y) = dp.scale_data(train, val, test, features)

            # Giá thực tế của test set (chưa scale) — dùng để đánh giá
            y_test_real = test[target].values
            y_val_real  = val[target].values
            test_dates  = test["Date"].astype(str).str[:10].tolist()

            # Gửi giá thực tế về UI để vẽ đường thực tế trên biểu đồ
            send("actual",
                 dates=test_dates,
                 values=[round(float(v)) for v in y_test_real])
            log("      ✓ Xong")

            # ────────────────────────────────────────────────
            #  STEP 5 — HUẤN LUYỆN CÁC MODEL
            # ────────────────────────────────────────────────
            log("[5/5] Huấn luyện: Ridge → RF → BiGRU → Ensemble...")

            # Lưu trữ kết quả train
            results        = {}   # model_id → dict metrics
            predictions    = {}   # model_id → {dates, values} trên TEST
            trained_models = {}   # model_id → fitted model object
            val_preds      = {}   # model_id → np.ndarray pred trên VAL
                                  # ↑ dùng để tìm Ensemble weights mà không nhìn test

            # ── Scale riêng cho Ridge (dự báo Return) ────
            # Ridge dùng StandardScaler + target là Return(T+1)
            # thay vì MinMaxScaler + target là Close(T+1)
            # → tránh Lag_1 dominant → model học pattern thật
            (X_train_r, y_train_r,
             X_val_r,   y_val_r,
             X_test_r,
             close_val, close_test,
             scaler_Xr, scaler_yr) = dp.scale_data_ridge(train, val, test)

            # ── Chuẩn bị sequences cho BiGRU ─────────────
            dl_ready = False
            if any(m in models for m in ["BiGRU", "Ensemble"]):
                Xs_tr, ys_tr = dp.make_sequences(X_train, y_train, seq_len)
                Xs_vl, ys_vl = dp.make_sequences(X_val,   y_val,   seq_len)
                Xs_te, _     = dp.make_sequences(X_test,  y_test,  seq_len)
                y_te_seq     = y_test_real[seq_len:]
                td_seq       = test_dates[seq_len:]
                dl_ready     = len(Xs_tr) >= 10
                if not dl_ready:
                    log("      ⚠ Dữ liệu quá ít cho BiGRU, bỏ qua DL models.")

            # ════════════════════════════════════════════════
            #  MODEL 1 — RIDGE REGRESSION + OPTUNA
            #
            #  Target: Return(T+1) = (Close(T+1)-Close(T))/Close(T)
            #  → Sau predict: Close_pred = Close(T) × (1 + Return_pred)
            #
            #  Tại sao đổi target?
            #    Ridge cũ dự báo Close(T+1) → Lag_1=Close(T-1) dominant
            #    → ŷ(T) ≈ Close(T-1) → đường bị lệch T+1
            #    Ridge mới dự báo Return → Lag_1 không còn dominant
            #    → model buộc phải học pattern thực sự của thị trường
            # ════════════════════════════════════════════════
            if "Ridge(Optuna)" in models:
                mid = "Ridge(Optuna)"
                log(f"\n  ┌─ {mid}  [target: Return(T+1) → convert về Close]")
                log(f"  │  Optuna tìm alpha tối ưu ({n_trials} trials)...")
                progress(mid, 5)
                try:
                    model, best_alpha = mi.train_ridge_optuna(
                        X_train_r, y_train_r, X_val_r, y_val_r,
                        n_trials=n_trials,
                        send_log=lambda m: log(f"  │  {m}"),
                    )

                    # Predict Return trên TEST rồi convert về giá Close
                    return_pred_test = scaler_yr.inverse_transform(
                        model.predict(X_test_r).reshape(-1, 1)
                    ).ravel()
                    pred = dp.return_to_price(return_pred_test, close_test)

                    # Predict Return trên VAL rồi convert về giá Close
                    return_pred_val = scaler_yr.inverse_transform(
                        model.predict(X_val_r).reshape(-1, 1)
                    ).ravel()
                    vp = dp.return_to_price(return_pred_val, close_val)

                    r = _make_result(mid, y_test_real, pred, test_dates)
                    results[mid]        = r
                    predictions[mid]    = {"dates": test_dates,
                                           "values": [round(float(v)) for v in pred]}
                    trained_models[mid] = model
                    val_preds[mid]      = vp

                    progress(mid, 100)
                    log(f"  └─ ✓ alpha={best_alpha:.4f}  {_fmt_metrics(r)}")

                except Exception as e:
                    log(f"  └─ ⚠ Lỗi: {e}")

            # ════════════════════════════════════════════════
            #  MODEL 2 — RANDOM FOREST + OPTUNA
            #  Tìm 4 hyperparameters: n_estimators, max_depth,
            #  min_samples_leaf, max_features
            #  max_depth giới hạn ≤ 15 để tránh overfit
            # ════════════════════════════════════════════════
            if "RF(Optuna)" in models:
                mid = "RF(Optuna)"
                log(f"\n  ┌─ {mid}")
                log(f"  │  Optuna tìm hyperparameters ({n_trials} trials)...")
                progress(mid, 5)
                try:
                    model, bp = mi.train_rf_optuna(
                        X_train, y_train, X_val, y_val,
                        n_trials=n_trials,
                        send_log=lambda m: log(f"  │  {m}"),
                    )

                    pred = scaler_y.inverse_transform(
                        model.predict(X_test).reshape(-1, 1)
                    ).ravel()

                    vp = scaler_y.inverse_transform(
                        model.predict(X_val).reshape(-1, 1)
                    ).ravel()

                    r = _make_result(mid, y_test_real, pred, test_dates)
                    results[mid]        = r
                    predictions[mid]    = {"dates": test_dates,
                                           "values": [round(float(v)) for v in pred]}
                    trained_models[mid] = model
                    val_preds[mid]      = vp

                    progress(mid, 100)
                    log(f"  └─ ✓ depth={bp['max_depth']}  "
                        f"leaf={bp['min_samples_leaf']}  {_fmt_metrics(r)}")

                except Exception as e:
                    log(f"  └─ ⚠ Lỗi: {e}")

            # ════════════════════════════════════════════════
            #  MODEL 3 — BIDIRECTIONAL GRU
            #  Đọc chuỗi theo 2 chiều (forward + backward)
            #  Chống overfit: Dropout(0.3) + L2 + recurrent_dropout
            #  EarlyStopping(patience=10) + ReduceLROnPlateau
            # ════════════════════════════════════════════════
            if "BiGRU" in models and dl_ready:
                mid = "BiGRU"
                log(f"\n  ┌─ {mid}")
                log(f"  │  Bidirectional GRU  (seq_len={seq_len}, epochs={epochs})")
                progress(mid, 3)
                try:
                    model, hist = mi.train_bigru(
                        Xs_tr, ys_tr, Xs_vl, ys_vl,
                        epochs=epochs,
                        send_log=lambda m: log(f"  │  {m}"),
                        send_progress=lambda p: progress(mid, p),
                    )

                    # Gửi loss curve về UI để vẽ biểu đồ training history
                    send("training_history",
                         model=mid,
                         train_loss=hist.history.get("loss", []),
                         val_loss=hist.history.get("val_loss", []),
                         stopped_epoch=len(hist.history.get("loss", [])))

                    # Predict trên TEST (căn chỉnh với seq_len offset)
                    pred = scaler_y.inverse_transform(
                        model.predict(Xs_te, verbose=0)
                    ).ravel()

                    # Predict trên VAL (dùng cho Ensemble)
                    vp = scaler_y.inverse_transform(
                        model.predict(Xs_vl, verbose=0)
                    ).ravel()

                    r = _make_result(mid, y_te_seq, pred, td_seq)
                    results[mid]        = r
                    predictions[mid]    = {"dates": td_seq,
                                           "values": [round(float(v)) for v in pred]}
                    trained_models[mid] = model
                    val_preds[mid]      = vp

                    progress(mid, 100)
                    log(f"  └─ ✓ {_fmt_metrics(r)}")

                except Exception as e:
                    log(f"  └─ ⚠ Lỗi: {e}")

            # ════════════════════════════════════════════════
            #  MODEL 4 — ENSEMBLE (Weighted Average)
            #
            #  Quy trình không leak:
            #    1. Tìm trọng số tối ưu trên VAL predictions
            #       → Optuna minimize RMSE trên val set
            #    2. Apply trọng số đó lên TEST predictions
            #       → Test không tham gia tìm weights
            #
            #  Yêu cầu: ít nhất 2 model đã train thành công
            # ════════════════════════════════════════════════
            if "Ensemble" in models and len(val_preds) >= 2:
                mid = "Ensemble"
                log(f"\n  ┌─ {mid}")
                log(f"  │  Tìm trọng số tối ưu trên VAL set (không dùng test)...")
                progress(mid, 5)
                try:
                    # ── Bước 1: Align val predictions về cùng độ dài ──
                    # BiGRU val pred ngắn hơn Ridge/RF đúng seq_len phiên
                    min_len_val  = min(len(v) for v in val_preds.values())
                    val_preds_al = {k: v[-min_len_val:] for k, v in val_preds.items()}
                    y_val_al     = y_val_real[-min_len_val:]

                    # ── Bước 2: Optuna tìm trọng số tối ưu trên VAL ───
                    weights = mi.find_optimal_ensemble_weights(
                        val_preds_al,
                        y_val_al,
                        send_log=lambda m: log(f"  │  {m}"),
                    )
                    progress(mid, 70)
                    log(f"  │  Weights tìm được: {weights}")

                    # ── Bước 3: Apply weights lên TEST predictions ─────
                    # Chỉ dùng các model có trong weights (đã train thành công)
                    test_preds_al = {
                        k: np.array(predictions[k]["values"])
                        for k in weights
                        if k in predictions
                    }
                    min_len_test  = min(len(v) for v in test_preds_al.values())
                    test_preds_al = {k: v[-min_len_test:]
                                     for k, v in test_preds_al.items()}
                    y_true_test   = y_test_real[-min_len_test:]
                    ens_dates     = test_dates[-min_len_test:]

                    # ── Bước 4: Tính Ensemble prediction ──────────────
                    ens_pred = mi.ensemble_predict(test_preds_al, weights)

                    r = _make_result(mid, y_true_test,
                                     ens_pred.astype(float), ens_dates)
                    results[mid]     = r
                    predictions[mid] = {"dates": ens_dates,
                                        "values": [round(float(v)) for v in ens_pred]}

                    progress(mid, 100)
                    log(f"  └─ ✓ {_fmt_metrics(r)}")

                except Exception as e:
                    log(f"  └─ ⚠ Lỗi: {e}\n{traceback.format_exc()}")

            # ────────────────────────────────────────────────
            #  STEP 6 — TỔNG KẾT & GỬI KẾT QUẢ VỀ UI
            # ────────────────────────────────────────────────
            results_list = list(results.values())

            # Gửi toàn bộ kết quả + predictions để UI vẽ biểu đồ
            send("done", results=results_list, predictions=predictions)

            log("\n" + "━" * 48)
            log("✅ Huấn luyện hoàn tất!")

            if results_list:
                # Tìm model tốt nhất theo RMSE
                best = min(results_list, key=lambda r: r["rmse"])
                log(f"   🏆 Tốt nhất: {best['model']}  "
                    f"RMSE={best['rmse']:.0f}  R²={best['r2']:.4f}")

        except Exception:
            log("❌ Lỗi không mong muốn:\n" + traceback.format_exc())
        finally:
            # Gửi None để báo hiệu stream kết thúc
            q.put(None)

    # Chạy pipeline trong thread daemon (tự tắt khi server tắt)
    threading.Thread(target=run, daemon=True).start()

    return _stream_response(q)


# ════════════════════════════════════════════════════════════════
#  PHẦN 8 — ENDPOINT: CRAWL DỮ LIỆU (Server-Sent Events)
# ════════════════════════════════════════════════════════════════

@app.route("/api/crawl", methods=["POST"])
def api_crawl():
    """
    Tải dữ liệu lịch sử giá từ Yahoo Finance cho các mã VN30.
    Stream log tiến độ realtime về UI qua SSE.

    Request body (JSON):
      tickers : list   Danh sách mã (mặc định: tất cả VN30)
      years   : int    Số năm lấy về (mặc định 5)

    SSE events:
      log            → {msg}                    Log text
      crawl_progress → {index, total, ticker, status, rows?}
      crawl_done     → {success, failed}        Tổng kết

    File lưu ra:
      data/VN30_Data/<TICKER>/<TICKER>_historical.csv  ← toàn bộ
      data/VN30_Data/<TICKER>/<TICKER>_<N>y.csv        ← N năm gần nhất
    """
    body    = request.json or {}
    tickers = body.get("tickers", config.VN30_TICKERS)
    years   = int(body.get("years", 5))

    q = queue.Queue()

    def send(event_type: str, **kwargs):
        q.put(json.dumps({"type": event_type, **kwargs}))

    def log(msg: str):
        send("log", msg=msg)

    def run():
        try:
            import pathlib
            from datetime import datetime, timedelta

            import yfinance as yf

            # ── Tính khoảng thời gian cần crawl ──────────
            end_date   = datetime.now()
            start_date = end_date - timedelta(days=365 * years)
            start_str  = start_date.strftime("%Y-%m-%d")
            end_str    = end_date.strftime("%Y-%m-%d")

            base_dir = pathlib.Path(config.DATA_DIR)

            log(f"📡 Crawl {len(tickers)} mã VN30 từ Yahoo Finance")
            log(f"   Khoảng thời gian : {start_str} → {end_str}  ({years} năm)")
            log(f"   Lưu vào          : {base_dir.resolve()}")
            log("─" * 48)

            # Tạo thư mục cho từng mã nếu chưa có
            for t in tickers:
                (base_dir / t).mkdir(parents=True, exist_ok=True)

            success = []   # Mã tải thành công
            failed  = []   # Mã tải thất bại

            for i, ticker in enumerate(tickers):
                try:
                    send("crawl_progress", index=i, total=len(tickers),
                         ticker=ticker, status="downloading")
                    log(f"  [{i+1:02d}/{len(tickers)}] {ticker} đang tải...")

                    # ── Tải từ Yahoo Finance ──────────────
                    # Mã VN30 trên Yahoo có suffix .VN (vd: VCB.VN)
                    df = yf.download(
                        f"{ticker}.VN",
                        start=start_str,
                        end=end_str,
                        progress=False,
                        auto_adjust=True,   # Tự điều chỉnh giá khi có cổ tức / tách cổ phiếu
                    )

                    if df.empty:
                        raise ValueError("Không có dữ liệu trả về từ Yahoo Finance")

                    # ── Chuẩn hóa cột ────────────────────
                    # yfinance đôi khi trả về MultiIndex columns
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = df.columns.get_level_values(0)

                    df = df.reset_index()
                    df.columns = [c.strip() for c in df.columns]

                    # Đổi tên cột về chuẩn của data_processing.py
                    col_map = {
                        "Datetime": "Date", "datetime": "Date",
                        "Open": "Open", "High": "High",
                        "Low":  "Low",  "Close": "Close",
                        "Volume": "Volume",
                    }
                    df = df.rename(columns={k: v for k, v in col_map.items()
                                            if k in df.columns})

                    # Format ngày và chỉ giữ cột OHLCV
                    df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")
                    keep = [c for c in ["Date", "Open", "High", "Low",
                                        "Close", "Volume"] if c in df.columns]
                    df = df[keep].dropna(subset=["Close"])

                    # ── Lưu file ─────────────────────────
                    # historical: toàn bộ dữ liệu từ đầu đến nay
                    # Ny:         N năm gần nhất (khớp với tham số years)
                    hist_path = base_dir / ticker / f"{ticker}_historical.csv"
                    year_path = base_dir / ticker / f"{ticker}_{years}y.csv"
                    df.to_csv(hist_path, index=False)
                    df.to_csv(year_path, index=False)

                    rows = len(df)
                    last = df["Date"].iloc[-1] if rows else "?"
                    success.append(ticker)

                    send("crawl_progress", index=i + 1, total=len(tickers),
                         ticker=ticker, status="done", rows=rows)
                    log(f"         ✓ {rows} phiên  (đến {last})")

                except Exception as e:
                    failed.append(ticker)
                    send("crawl_progress", index=i + 1, total=len(tickers),
                         ticker=ticker, status="error")
                    log(f"         ⚠ Lỗi: {e}")

            # ── Tổng kết ─────────────────────────────────
            log("─" * 48)
            log(f"✅ Hoàn tất!  Thành công: {len(success)}/{len(tickers)}")
            if failed:
                log(f"   ❌ Thất bại: {', '.join(failed)}")

            send("crawl_done", success=success, failed=failed)

        except ImportError:
            log("❌ Chưa cài yfinance — chạy lệnh: pip install yfinance")
            send("crawl_done", success=[], failed=tickers)
        except Exception:
            log("❌ Lỗi không mong muốn:\n" + traceback.format_exc())
            send("crawl_done", success=[], failed=tickers)
        finally:
            q.put(None)

    threading.Thread(target=run, daemon=True).start()

    return _stream_response(q)


# ════════════════════════════════════════════════════════════════
#  PHẦN 9 — KHỞI ĐỘNG SERVER
# ════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "═" * 52)
    print("  🚀  VN30 Stock Prediction Dashboard")
    print("       http://localhost:5000")
    print("═" * 52 + "\n")

    # debug=False vì dùng threading (debug mode không tương thích tốt)
    # threaded=True cho phép nhiều request đồng thời
    app.run(debug=False, threaded=True, port=5000)
