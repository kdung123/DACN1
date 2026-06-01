#!/usr/bin/env python3
# ============================================================
# main.py  –  Pipeline dự báo giá cổ phiếu VN30
#
#  Cấu trúc dữ liệu mong đợi:
#    VN30/VN30_Data/<TICKER>/<TICKER>_historical.csv
#    Cột: Date, Open, High, Low, Close, Volume
#
#  Cách dùng:
#    python main.py                         # Chạy tất cả 30 mã
#    python main.py --ticker VCB            # Chỉ 1 mã
#    python main.py --ticker VCB --mode 2y  # Dùng file _2y.csv
#    python main.py --no-dl                 # Bỏ LSTM+Attention / BiGRU
#    python main.py --eda-only              # Chỉ vẽ EDA, không huấn luyện
# ============================================================

import os, argparse, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import config
import data_processing as dp
import models as m
import evaluation as ev


def run_pipeline(ticker: str, mode: str = config.DATA_MODE,
                 use_dl: bool = True, eda_only: bool = False):

    print(f"\n{'═'*62}")
    print(f"  {ticker}   (file: {ticker}_{mode}.csv)")
    print(f"{'═'*62}")

    # ── 1. Load ──────────────────────────────────────────────
    print("\n[1] Load dữ liệu")
    df_raw = dp.load_stock_data(ticker, mode)
    df_pre = dp.preprocess(df_raw, ticker)

    # ── 2. EDA plot ──────────────────────────────────────────
    print("\n[2] EDA – Vẽ lịch sử giá")
    ev.plot_close_history(df_pre, ticker)

    if eda_only:
        return None

    # ── 3. Feature Engineering ───────────────────────────────
    print("\n[3] Feature Engineering")
    df = dp.feature_engineering(df_pre)

    # ── 4. Split ─────────────────────────────────────────────
    print("\n[4] Chia Train / Val / Test")
    train, val, test = dp.split_data(df)

    features    = dp.get_feature_columns(df)
    target      = config.TARGET_COLUMN
    y_test_real = test[target].values       # Giá thực tế (chưa scale)
    test_dates  = test[config.DATE_COLUMN].values

    # ── 5. Scale ─────────────────────────────────────────────
    print("\n[5] Chuẩn hóa MinMaxScaler")
    (X_train, y_train,
     X_val,   y_val,
     X_test,  y_test,
     scaler_X, scaler_y) = dp.scale_data(train, val, test, features)

    results     = []
    predictions = {}

    # ── 6a. Auto ARIMA + Walk-forward ────────────────────────
    print("\n[6a] Auto ARIMA (walk-forward)")
    try:
        arima_series = pd.concat([train, val])[target]
        fitted_arima, order = m.train_auto_arima(arima_series)
        arima_pred = m.predict_arima_walkforward(
            arima_series, y_test_real, order
        )
        results.append(ev.evaluate(y_test_real, arima_pred, "AutoARIMA"))
        predictions["AutoARIMA"] = arima_pred
    except Exception as e:
        print(f"  ⚠ AutoARIMA lỗi: {e}")

    # ── 6b. Ridge Regression + Optuna ────────────────────────
    print("\n[6b] Ridge Regression + Optuna")
    try:
        ridge_model, best_alpha = m.train_ridge_optuna(
            X_train, y_train, X_val, y_val
        )
        ridge_pred = scaler_y.inverse_transform(
            ridge_model.predict(X_test).reshape(-1, 1)
        ).ravel()
        results.append(ev.evaluate(y_test_real, ridge_pred, "Ridge(Optuna)"))
        predictions["Ridge(Optuna)"] = ridge_pred
    except Exception as e:
        print(f"  ⚠ Ridge lỗi: {e}")

    # ── 6c. Random Forest + Optuna ───────────────────────────
    print("\n[6c] Random Forest + Optuna")
    try:
        rf_model, best_params = m.train_rf_optuna(
            X_train, y_train, X_val, y_val
        )
        rf_pred = scaler_y.inverse_transform(
            rf_model.predict(X_test).reshape(-1, 1)
        ).ravel()
        results.append(ev.evaluate(y_test_real, rf_pred, "RF(Optuna)"))
        predictions["RF(Optuna)"] = rf_pred
    except Exception as e:
        print(f"  ⚠ RandomForest lỗi: {e}")

    # ── 6d/e. Deep Learning (LSTM+Attention & BiGRU) ─────────
    if use_dl:
        seq = config.SEQUENCE_LEN

        Xs_tr, ys_tr = dp.make_sequences(X_train, y_train, seq)
        Xs_vl, ys_vl = dp.make_sequences(X_val,   y_val,   seq)
        Xs_te, _     = dp.make_sequences(X_test,  y_test,  seq)
        y_te_seq     = y_test_real[seq:]    # căn chỉnh với sequence
        td_seq       = test_dates[seq:]

        if len(Xs_tr) < 10:
            print("  ⚠ Dữ liệu quá ít cho DL, bỏ qua.")
            use_dl = False

    if use_dl:
        print("\n[6d] LSTM + Attention")
        try:
            lstm_model = m.train_lstm_attention(Xs_tr, ys_tr, Xs_vl, ys_vl)
            m.save_model(lstm_model, f"lstm_attn_{ticker}")
            lstm_pred = scaler_y.inverse_transform(
                lstm_model.predict(Xs_te, verbose=0)
            ).ravel()
            results.append(ev.evaluate(y_te_seq, lstm_pred, "LSTM+Attention"))
            predictions["LSTM+Attention"] = lstm_pred
        except Exception as e:
            print(f"  ⚠ LSTM+Attention lỗi: {e}")

        print("\n[6e] Bidirectional GRU")
        try:
            bigru_model = m.train_bigru(Xs_tr, ys_tr, Xs_vl, ys_vl)
            m.save_model(bigru_model, f"bigru_{ticker}")
            bigru_pred = scaler_y.inverse_transform(
                bigru_model.predict(Xs_te, verbose=0)
            ).ravel()
            results.append(ev.evaluate(y_te_seq, bigru_pred, "BiGRU"))
            predictions["BiGRU"] = bigru_pred
        except Exception as e:
            print(f"  ⚠ BiGRU lỗi: {e}")

    # ── 6f. Ensemble (Optuna weights) ────────────────────────
    if len(predictions) >= 2:
        print("\n[6f] Ensemble (Optuna tìm trọng số)")
        try:
            # Căn chỉnh tất cả predictions về cùng độ dài test nhỏ nhất
            min_len = min(len(p) for p in predictions.values())
            preds_aligned_ens = {k: v[-min_len:] for k, v in predictions.items()}
            y_true_ens = y_test_real[-min_len:]

            best_weights = m.find_optimal_ensemble_weights(
                preds_aligned_ens, y_true_ens
            )
            ens_pred = m.ensemble_predict(preds_aligned_ens, best_weights)
            results.append(ev.evaluate(y_true_ens, ens_pred, "Ensemble"))
            predictions["Ensemble"] = ens_pred
        except Exception as e:
            print(f"  ⚠ Ensemble lỗi: {e}")

    # ── 7. So sánh ───────────────────────────────────────────
    print("\n[7] Tổng hợp")
    if not results:
        print("  ⚠ Không có kết quả nào để so sánh.")
        return None
    df_metrics = ev.compare_models(results)
    ev.save_results_csv(df_metrics, ticker)

    # ── 8. Plots ─────────────────────────────────────────────
    print("\n[8] Đồ thị")
    # Căn chỉnh tất cả predictions về cùng kích thước test_dates
    min_len_plot = min(len(test_dates), min(len(p) for p in predictions.values()))
    preds_plot = {k: v[-min_len_plot:] for k, v in predictions.items()}
    ev.plot_predictions(
        test_dates[-min_len_plot:],
        y_test_real[-min_len_plot:],
        preds_plot,
        ticker,
    )
    ev.plot_metrics_comparison(df_metrics, ticker)

    print(f"\n  ✅ Xong {ticker}")
    return df_metrics


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dự báo giá cổ phiếu VN30")
    parser.add_argument("--ticker",   type=str, default=None,
                        help="Mã cổ phiếu (vd: VCB). Mặc định: chạy tất cả")
    parser.add_argument("--mode",     type=str, default=config.DATA_MODE,
                        choices=["2y", "5y", "historical"],
                        help="Loại file dữ liệu (mặc định: historical)")
    parser.add_argument("--no-dl",    action="store_true",
                        help="Bỏ qua LSTM+Attention / BiGRU")
    parser.add_argument("--eda-only", action="store_true",
                        help="Chỉ vẽ EDA, không huấn luyện mô hình")
    args = parser.parse_args()

    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    os.makedirs(config.MODELS_DIR,  exist_ok=True)

    tickers = [args.ticker.upper()] if args.ticker else config.VN30_TICKERS

    all_results = {}
    for t in tickers:
        try:
            df_m = run_pipeline(
                t,
                mode=args.mode,
                use_dl=not args.no_dl,
                eda_only=args.eda_only,
            )
            if df_m is not None:
                all_results[t] = df_m
        except Exception as e:
            print(f"\n  ❌ {t}: {e}")

    # Tổng hợp tất cả mã
    if all_results:
        summary = pd.concat(
            {k: v for k, v in all_results.items()},
            names=["Ticker", "Model"],
        )
        path = os.path.join(config.RESULTS_DIR, "summary_all.csv")
        summary.to_csv(path)
        print(f"\n{'═'*62}")
        print(f"  TỔNG HỢP tất cả mã → {path}")
        print("  Mô hình tốt nhất mỗi mã (RMSE thấp nhất):")
        best = summary.groupby("Ticker")["RMSE"].idxmin()
        for ticker, idx in best.items():
            row = summary.loc[idx]
            print(f"    {ticker:5s}  {idx[1]:<22}  RMSE={row['RMSE']:.2f}")
        print(f"{'═'*62}\n")
