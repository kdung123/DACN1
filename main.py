# ╔══════════════════════════════════════════════════════════════╗
# ║                         main.py                             ║
# ║          Pipeline dự báo giá cổ phiếu VN30 (CLI)           ║
# ╠══════════════════════════════════════════════════════════════╣
# ║  Cách dùng:                                                 ║
# ║    python main.py                     # Chạy tất cả 30 mã  ║
# ║    python main.py --ticker VCB        # Chỉ 1 mã            ║
# ║    python main.py --no-dl             # Bỏ qua BiGRU        ║
# ║    python main.py --eda-only          # Chỉ vẽ EDA          ║
# ╚══════════════════════════════════════════════════════════════╝

import os
import argparse
import warnings

import numpy as np
import pandas as pd

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

import config
import data_processing as dp
import evaluation as ev
import models as m


# ════════════════════════════════════════════════════════════════
#  PIPELINE CHÍNH
# ════════════════════════════════════════════════════════════════

def run_pipeline(ticker: str,
                 mode: str    = config.DATA_MODE,
                 use_dl: bool = True,
                 eda_only: bool = False):
    """
    Chạy toàn bộ pipeline dự báo cho một mã cổ phiếu:
      1. Load & tiền xử lý
      2. EDA (vẽ lịch sử giá)
      3. Feature Engineering
      4. Split → Scale → (Sequence nếu dùng DL)
      5. Train: Ridge | RF | BiGRU | Ensemble
      6. Đánh giá & vẽ biểu đồ

    Parameters
    ----------
    ticker   : str    Mã cổ phiếu (vd: "VCB")
    mode     : str    Loại file dữ liệu ("2y" | "5y" | "historical")
    use_dl   : bool   Có train BiGRU không
    eda_only : bool   Chỉ vẽ EDA, bỏ qua train

    Returns
    -------
    pd.DataFrame | None   Bảng metrics (None nếu eda_only hoặc lỗi)
    """
    print(f"\n{'═' * 64}")
    print(f"  {ticker}   ·   {ticker}_{mode}.csv")
    print(f"{'═' * 64}")

    # ── [1] Load & tiền xử lý ────────────────────────────────
    print("\n[1/6] Load & tiền xử lý")
    df_raw = dp.load_stock_data(ticker, mode)
    df_pre = dp.preprocess(df_raw, ticker)

    # ── [2] EDA ───────────────────────────────────────────────
    print("\n[2/6] EDA — Vẽ lịch sử giá")
    ev.plot_close_history(df_pre, ticker)
    if eda_only:
        return None

    # ── [3] Feature Engineering ───────────────────────────────
    print("\n[3/6] Feature Engineering")
    df = dp.feature_engineering(df_pre)

    # ── [4] Split & Scale ─────────────────────────────────────
    print("\n[4/6] Chia Train / Val / Test  (70 / 15 / 15)")
    train, val, test = dp.split_data(df)

    features    = dp.get_feature_columns(df)
    target      = config.TARGET_COLUMN
    y_test_real = test[target].values
    test_dates  = test[config.DATE_COLUMN].values

    print("\n[5/6] Chuẩn hóa MinMaxScaler")
    (X_train, y_train,
     X_val,   y_val,
     X_test,  y_test,
     scaler_X, scaler_y) = dp.scale_data(train, val, test, features)

    results      = []
    predictions  = {}   # pred trên TEST  — dùng để đánh giá & vẽ biểu đồ
    preds_val    = {}   # pred trên VAL   — dùng để tìm trọng số Ensemble
    y_val_real   = val[target].values

    # ── [5a] Ridge Regression ─────────────────────────────────
    print("\n  ▶ Ridge Regression + Optuna")
    try:
        ridge_model, best_alpha = m.train_ridge_optuna(
            X_train, y_train, X_val, y_val
        )
        ridge_pred = scaler_y.inverse_transform(
            ridge_model.predict(X_test).reshape(-1, 1)
        ).ravel()
        # Val pred — dùng để tìm trọng số Ensemble (không đụng test)
        ridge_pred_val = scaler_y.inverse_transform(
            ridge_model.predict(X_val).reshape(-1, 1)
        ).ravel()

        results.append(ev.evaluate(y_test_real, ridge_pred, "Ridge(Optuna)"))
        predictions["Ridge(Optuna)"] = ridge_pred
        preds_val["Ridge(Optuna)"]   = ridge_pred_val
    except Exception as e:
        print(f"  ⚠  Ridge lỗi: {e}")

    # ── [5b] Random Forest ────────────────────────────────────
    print("\n  ▶ Random Forest + Optuna")
    try:
        rf_model, best_params = m.train_rf_optuna(
            X_train, y_train, X_val, y_val
        )
        rf_pred = scaler_y.inverse_transform(
            rf_model.predict(X_test).reshape(-1, 1)
        ).ravel()
        rf_pred_val = scaler_y.inverse_transform(
            rf_model.predict(X_val).reshape(-1, 1)
        ).ravel()

        results.append(ev.evaluate(y_test_real, rf_pred, "RF(Optuna)"))
        predictions["RF(Optuna)"] = rf_pred
        preds_val["RF(Optuna)"]   = rf_pred_val
    except Exception as e:
        print(f"  ⚠  Random Forest lỗi: {e}")

    # ── [5c] BiGRU ────────────────────────────────────────────
    if use_dl:
        print("\n  ▶ Bidirectional GRU")
        seq = config.SEQUENCE_LEN

        Xs_tr, ys_tr = dp.make_sequences(X_train, y_train, seq)
        Xs_vl, ys_vl = dp.make_sequences(X_val,   y_val,   seq)
        Xs_te, _     = dp.make_sequences(X_test,  y_test,  seq)
        y_te_seq     = y_test_real[seq:]   # align với sequence offset
        td_seq       = test_dates[seq:]

        if len(Xs_tr) < 10:
            print("  ⚠  Dữ liệu quá ít cho BiGRU, bỏ qua.")
            use_dl = False

    if use_dl:
        try:
            bigru_model, bigru_hist = m.train_bigru(Xs_tr, ys_tr, Xs_vl, ys_vl)
            m.save_model(bigru_model, f"bigru_{ticker}")
            ev.plot_training_history({"BiGRU": bigru_hist}, ticker)

            bigru_pred = scaler_y.inverse_transform(
                bigru_model.predict(Xs_te, verbose=0)
            ).ravel()
            bigru_pred_val = scaler_y.inverse_transform(
                bigru_model.predict(Xs_vl, verbose=0)
            ).ravel()

            results.append(ev.evaluate(y_te_seq, bigru_pred, "BiGRU"))
            predictions["BiGRU"] = bigru_pred
            preds_val["BiGRU"]   = bigru_pred_val
        except Exception as e:
            print(f"  ⚠  BiGRU lỗi: {e}")

    # ── [5d] Ensemble ─────────────────────────────────────────
    # Trọng số tìm trên VAL set → apply lên TEST set
    # → tránh data leakage (không dùng nhãn test để tune weights)
    if len(preds_val) >= 2:
        print("\n  ▶ Ensemble — Optuna tìm trọng số trên VAL")
        try:
            # Align val predictions về cùng độ dài
            min_len_val  = min(len(p) for p in preds_val.values())
            preds_val_al = {k: v[-min_len_val:] for k, v in preds_val.items()}
            y_val_al     = y_val_real[-min_len_val:]

            # Tìm trọng số tối ưu trên VAL (không nhìn test)
            best_weights = m.find_optimal_ensemble_weights(
                preds_val_al, y_val_al
            )

            # Apply trọng số lên TEST predictions
            preds_test_al = {
                k: predictions[k]
                for k in best_weights if k in predictions
            }
            min_len_test  = min(len(v) for v in preds_test_al.values())
            preds_test_al = {k: v[-min_len_test:] for k, v in preds_test_al.items()}
            y_true_test   = y_test_real[-min_len_test:]

            ens_pred = m.ensemble_predict(preds_test_al, best_weights)

            results.append(ev.evaluate(y_true_test, ens_pred, "Ensemble"))
            predictions["Ensemble"] = ens_pred
        except Exception as e:
            print(f"  ⚠  Ensemble lỗi: {e}")

    # ── [6] Đánh giá & Biểu đồ ───────────────────────────────
    print(f"\n[6/6] Tổng hợp kết quả")
    if not results:
        print("  ⚠  Không có model nào chạy thành công.")
        return None

    df_metrics = ev.compare_models(results)
    ev.save_results_csv(df_metrics, ticker)

    # Align predictions về cùng độ dài để vẽ biểu đồ
    min_len_plot = min(len(test_dates),
                       min(len(p) for p in predictions.values()))
    ev.plot_predictions(
        test_dates[-min_len_plot:],
        y_test_real[-min_len_plot:],
        {k: v[-min_len_plot:] for k, v in predictions.items()},
        ticker,
    )
    ev.plot_metrics_comparison(df_metrics, ticker)

    print(f"\n  ✅ Xong {ticker}")
    return df_metrics


# ════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Dự báo giá cổ phiếu VN30  |  Ridge · RF · BiGRU · Ensemble"
    )
    parser.add_argument(
        "--ticker",   type=str, default=None,
        help="Mã cổ phiếu (vd: VCB). Mặc định: chạy tất cả 30 mã"
    )
    parser.add_argument(
        "--mode",     type=str, default=config.DATA_MODE,
        choices=["2y", "5y", "historical"],
        help="Loại file dữ liệu (mặc định: historical)"
    )
    parser.add_argument(
        "--no-dl",    action="store_true",
        help="Bỏ qua BiGRU (chạy chỉ Ridge + RF + Ensemble)"
    )
    parser.add_argument(
        "--eda-only", action="store_true",
        help="Chỉ vẽ EDA, không huấn luyện model"
    )
    args = parser.parse_args()

    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    os.makedirs(config.MODELS_DIR,  exist_ok=True)

    tickers = [args.ticker.upper()] if args.ticker else config.VN30_TICKERS

    # ── Chạy từng mã ─────────────────────────────────────────
    all_results = {}
    for t in tickers:
        try:
            df_m = run_pipeline(
                t,
                mode     = args.mode,
                use_dl   = not args.no_dl,
                eda_only = args.eda_only,
            )
            if df_m is not None:
                all_results[t] = df_m
        except Exception as e:
            print(f"\n  ❌ {t}: {e}")

    # ── Tổng hợp tất cả mã ───────────────────────────────────
    if all_results:
        summary = pd.concat(
            {k: v for k, v in all_results.items()},
            names=["Ticker", "Model"],
        )
        path = os.path.join(config.RESULTS_DIR, "summary_all.csv")
        summary.to_csv(path)

        print(f"\n{'═' * 64}")
        print(f"  TỔNG HỢP  →  {path}")
        print("  Model tốt nhất mỗi mã (RMSE thấp nhất):")
        best = summary.groupby("Ticker")["RMSE"].idxmin()
        for ticker, idx in best.items():
            row = summary.loc[idx]
            print(f"    {ticker:5s}  {idx[1]:<22}  RMSE={row['RMSE']:.0f}")
        print(f"{'═' * 64}\n")
