# ============================================================
# models.py  –  Tất cả mô hình dự báo (đã gộp & cải tiến)
#
#  1. auto_arima      — tự chọn (p,d,q) tốt nhất + walk-forward
#  2. optuna_ridge    — Ridge Regression + Optuna tìm alpha
#  3. optuna_rf       — Random Forest + Optuna tìm hyperparameter
#  4. lstm_attention  — LSTM + custom Attention layer
#  5. bigru           — Bidirectional GRU
#  6. ensemble        — Weighted average tối ưu bằng Optuna
# ============================================================

import os, warnings
import numpy as np
import pandas as pd
import optuna

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")

import config


# ════════════════════════════════════════════════════════════
#  1. AUTO ARIMA
# ════════════════════════════════════════════════════════════

def train_auto_arima(train_series: pd.Series, send_log=None):
    """
    Dùng pmdarima.auto_arima để tự chọn (p,d,q) tốt nhất theo AIC.
    Cải tiến so với ARIMA cố định (5,1,0): tự động tìm order phù hợp
    với từng mã cổ phiếu.
    """
    from pmdarima import auto_arima

    log = send_log or print
    log("    auto_arima: đang tìm (p,d,q) tốt nhất...")

    model = auto_arima(
        train_series,
        start_p=0, max_p=6,
        start_q=0, max_q=3,
        d=None,           # tự chọn d qua ADF test
        seasonal=False,
        information_criterion="aic",
        stepwise=True,
        error_action="ignore",
        suppress_warnings=True,
    )
    order = model.order
    log(f"    ✓ auto_arima chọn ARIMA{order}  AIC={model.aic():.1f}")
    return model, order


def predict_arima_walkforward(train_series: pd.Series,
                               y_test_real: np.ndarray,
                               order: tuple,
                               send_log=None,
                               send_progress=None):
    """
    Walk-forward validation: dự báo từng bước 1, update incremental
    bằng model.append() thay vì fit lại từ đầu → nhanh hơn O(n) lần.
    """
    from statsmodels.tsa.arima.model import ARIMA as _A

    log      = send_log      or print
    progress = send_progress or (lambda p: None)

    history = list(train_series.values)
    preds   = []
    n       = len(y_test_real)

    # Fit một lần duy nhất trên toàn bộ train
    fitted = _A(history, order=order).fit()

    for i in range(n):
        yhat = float(fitted.forecast(steps=1)[0])
        preds.append(round(yhat))

        # Update incremental — không fit lại từ đầu
        fitted = fitted.append([float(y_test_real[i])], refit=False)

        if i % max(1, n // 8) == 0:
            progress(int(i / n * 85) + 10)
            log(f"      walk-forward {i+1}/{n}...")

    return np.array(preds)


# ════════════════════════════════════════════════════════════
#  2. RIDGE REGRESSION + OPTUNA
# ════════════════════════════════════════════════════════════

def train_ridge_optuna(X_train, y_train, X_val, y_val,
                       n_trials=30, send_log=None):
    """
    Cải tiến từ LinearRegression: thêm L2 regularization (Ridge)
    và tự động tìm hệ số alpha tối ưu bằng Optuna.

    LinearRegression: loss = MSE
    Ridge:            loss = MSE + alpha × Σ(w²)
    """
    from sklearn.linear_model import Ridge
    from sklearn.metrics import mean_squared_error

    log = send_log or print
    log(f"    Optuna Ridge: {n_trials} trials tìm alpha...")

    def objective(trial):
        alpha = trial.suggest_float("alpha", 1e-3, 100, log=True)
        model = Ridge(alpha=alpha).fit(X_train, y_train)
        return float(mean_squared_error(y_val, model.predict(X_val)))

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_alpha = study.best_params["alpha"]
    log(f"    ✓ Ridge best alpha={best_alpha:.4f}  val_mse={study.best_value:.6f}")

    # Fit lại trên train+val với alpha tốt nhất
    final = Ridge(alpha=best_alpha).fit(
        np.concatenate([X_train, X_val]),
        np.concatenate([y_train, y_val])
    )
    return final, best_alpha


# ════════════════════════════════════════════════════════════
#  3. RANDOM FOREST + OPTUNA
# ════════════════════════════════════════════════════════════

def train_rf_optuna(X_train, y_train, X_val, y_val,
                    n_trials=20, send_log=None):
    """
    Cải tiến từ RandomForest cố định hyperparameter:
    Optuna tự động tìm n_estimators, max_depth, min_samples_leaf,
    max_features tối ưu.
    """
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_squared_error

    log = send_log or print
    log(f"    Optuna RF: {n_trials} trials tìm hyperparameters...")

    def objective(trial):
        params = {
            "n_estimators":     trial.suggest_int("n_estimators", 50, 300, step=50),
            "max_depth":        trial.suggest_int("max_depth", 5, 25),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features":     trial.suggest_categorical("max_features", ["sqrt", "log2", 0.5]),
        }
        model = RandomForestRegressor(**params, random_state=42, n_jobs=-1).fit(X_train, y_train)
        return float(mean_squared_error(y_val, model.predict(X_val)))

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    bp = study.best_params
    log(f"    ✓ RF best: n_est={bp['n_estimators']} depth={bp['max_depth']} "
        f"leaf={bp['min_samples_leaf']}  val_mse={study.best_value:.4f}")

    final = RandomForestRegressor(**bp, random_state=42, n_jobs=-1).fit(
        np.concatenate([X_train, X_val]),
        np.concatenate([y_train, y_val])
    )
    return final, bp


# ════════════════════════════════════════════════════════════
#  4. LSTM + ATTENTION LAYER (tự implement)
# ════════════════════════════════════════════════════════════

def build_lstm_attention(input_shape: tuple, units=(64, 32), dropout=0.2):
    """
    Cải tiến từ LSTM thuần: thêm custom Attention layer tự viết.

    Kiến trúc:
        Input → LSTM(64, return_seq=True) → Dropout → AttentionLayer
              → Dense(32, relu) → Dropout → Dense(1)

    Attention:
        score(t) = tanh(W·h(t) + b)     # alignment score
        alpha(t) = softmax(score)         # attention weight
        context  = Σ alpha(t) × h(t)     # weighted context vector
    """
    import tensorflow as tf

    Model         = tf.keras.Model
    LSTM          = tf.keras.layers.LSTM
    Dense         = tf.keras.layers.Dense
    Dropout       = tf.keras.layers.Dropout
    Layer         = tf.keras.layers.Layer
    KInput        = tf.keras.layers.Input

    class AttentionLayer(Layer):
        """Self-attention trên output LSTM — tự viết, không dùng thư viện."""
        def build(self, input_shape):
            self.W = self.add_weight(name="W", shape=(input_shape[-1], 1),
                                     initializer="glorot_uniform", trainable=True)
            self.b = self.add_weight(name="b", shape=(1,),
                                     initializer="zeros", trainable=True)
            super().build(input_shape)

        def call(self, x):
            score   = tf.nn.tanh(tf.matmul(x, self.W) + self.b)  # (batch, T, 1)
            alpha   = tf.nn.softmax(score, axis=1)                 # (batch, T, 1)
            context = tf.reduce_sum(alpha * x, axis=1)             # (batch, features)
            return context

        def get_config(self):
            return super().get_config()

    tf.random.set_seed(config.RANDOM_SEED)
    inp  = KInput(shape=input_shape)
    x    = LSTM(units[0], return_sequences=True)(inp)
    x    = Dropout(dropout)(x)
    x    = AttentionLayer()(x)
    x    = Dense(units[1], activation="relu")(x)
    x    = Dropout(dropout)(x)
    out  = Dense(1)(x)

    model = Model(inputs=inp, outputs=out, name="LSTM_Attention")
    model.compile(optimizer="adam", loss="mse")
    return model


def train_lstm_attention(Xs_tr, ys_tr, Xs_vl, ys_vl,
                         epochs=None, send_log=None, send_progress=None):
    import tensorflow as tf
    EarlyStopping      = tf.keras.callbacks.EarlyStopping
    ReduceLROnPlateau  = tf.keras.callbacks.ReduceLROnPlateau
    Callback           = tf.keras.callbacks.Callback

    epochs   = epochs or config.DL_EPOCHS
    log      = send_log      or print
    progress = send_progress or (lambda p: None)

    class CB(Callback):
        def on_epoch_end(self, epoch, logs=None):
            progress(min(int((epoch + 1) / epochs * 90) + 5, 95))
            lr = float(self.model.optimizer.learning_rate)
            log(f"      epoch {epoch+1}/{epochs}  "
                f"val_loss={logs.get('val_loss', 0):.5f}  lr={lr:.2e}")

    model = build_lstm_attention((Xs_tr.shape[1], Xs_tr.shape[2]))
    log(f"    LSTM+Attention: {model.count_params():,} params")

    hist = model.fit(
        Xs_tr, ys_tr,
        validation_data=(Xs_vl, ys_vl),
        epochs=epochs,
        batch_size=32,
        verbose=0,
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=config.DL_PATIENCE,
                          restore_best_weights=True, verbose=0),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                              patience=5, min_lr=1e-6, verbose=0),
            CB(),
        ],
    )
    log(f"    ✓ LSTM+Attention dừng ở epoch {len(hist.history['loss'])}/{epochs}")
    return model, hist


# ════════════════════════════════════════════════════════════
#  5. BIDIRECTIONAL GRU
# ════════════════════════════════════════════════════════════

def build_bigru(input_shape: tuple, units=(64, 32), dropout=0.2):
    """
    Cải tiến từ GRU đơn hướng: Bidirectional GRU đọc chuỗi
    theo cả 2 chiều (forward + backward), tăng khả năng học pattern.

    Kiến trúc:
        Input → Bidirectional(GRU(64, return_seq=True)) → Dropout
              → GRU(32, return_sequences=False) → Dropout → Dense(1)
    """
    import tensorflow as tf

    Sequential    = tf.keras.Sequential
    GRU           = tf.keras.layers.GRU
    Bidirectional = tf.keras.layers.Bidirectional
    Dense         = tf.keras.layers.Dense
    Dropout       = tf.keras.layers.Dropout
    KInput        = tf.keras.layers.Input

    tf.random.set_seed(config.RANDOM_SEED)
    model = Sequential([
        KInput(shape=input_shape),
        Bidirectional(GRU(units[0], return_sequences=True)),
        Dropout(dropout),
        GRU(units[1], return_sequences=False),
        Dropout(dropout),
        Dense(1),
    ], name="BiGRU")
    model.compile(optimizer="adam", loss="mse")
    return model


def train_bigru(Xs_tr, ys_tr, Xs_vl, ys_vl,
                epochs=None, send_log=None, send_progress=None):
    import tensorflow as tf
    EarlyStopping      = tf.keras.callbacks.EarlyStopping
    ReduceLROnPlateau  = tf.keras.callbacks.ReduceLROnPlateau
    Callback           = tf.keras.callbacks.Callback

    epochs   = epochs or config.DL_EPOCHS
    log      = send_log      or print
    progress = send_progress or (lambda p: None)

    class CB(Callback):
        def on_epoch_end(self, epoch, logs=None):
            progress(min(int((epoch + 1) / epochs * 90) + 5, 95))
            lr = float(self.model.optimizer.learning_rate)
            log(f"      epoch {epoch+1}/{epochs}  "
                f"val_loss={logs.get('val_loss', 0):.5f}  lr={lr:.2e}")

    model = build_bigru((Xs_tr.shape[1], Xs_tr.shape[2]))
    log(f"    BiGRU: {model.count_params():,} params")

    hist = model.fit(
        Xs_tr, ys_tr,
        validation_data=(Xs_vl, ys_vl),
        epochs=epochs,
        batch_size=32,
        verbose=0,
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=config.DL_PATIENCE,
                          restore_best_weights=True, verbose=0),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                              patience=5, min_lr=1e-6, verbose=0),
            CB(),
        ],
    )
    log(f"    ✓ BiGRU dừng ở epoch {len(hist.history['loss'])}/{epochs}")
    return model, hist


# ════════════════════════════════════════════════════════════
#  6. ENSEMBLE
# ════════════════════════════════════════════════════════════

def ensemble_predict(predictions: dict, weights: dict = None) -> np.ndarray:
    """
    Kết hợp dự báo nhiều model theo trọng số.
    Tự động căn chỉnh độ dài về min.
    weights=None → trọng số bằng nhau.
    Trả về float (round đến hàng đơn vị) thay vì int để giữ độ chính xác.
    """
    names   = list(predictions.keys())
    arrs    = [np.array(predictions[n]) for n in names]
    min_len = min(len(a) for a in arrs)
    arrs    = [a[-min_len:] for a in arrs]

    if weights is None:
        w = np.ones(len(arrs)) / len(arrs)
    else:
        w_raw = np.array([weights.get(n, 1.0) for n in names])
        w = w_raw / w_raw.sum()

    pred = sum(wi * a for wi, a in zip(w, arrs))
    return np.round(pred, 0)   # float, round đến hàng đơn vị


def find_optimal_ensemble_weights(predictions: dict,
                                   y_true: np.ndarray,
                                   n_trials: int = 100,
                                   send_log=None) -> dict:
    """
    Optuna tìm trọng số tối ưu minimize RMSE ensemble.
    Tăng n_trials lên 100 để search space 5 chiều đủ coverage.
    """
    log   = send_log or print
    names = list(predictions.keys())
    arrs  = [np.array(predictions[n]) for n in names]
    n     = min(len(a) for a in arrs + [y_true])
    arrs  = [a[-n:] for a in arrs]
    y     = y_true[-n:]

    def objective(trial):
        w = np.array([trial.suggest_float(nm, 0.0, 1.0) for nm in names])
        if w.sum() < 1e-6:
            return 1e9
        w /= w.sum()
        pred = sum(wi * a for wi, a in zip(w, arrs))
        return float(np.sqrt(np.mean((y - pred) ** 2)))

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    raw   = {n: study.best_params[n] for n in names}
    total = sum(raw.values())
    best  = {n: round(v / total, 3) for n, v in raw.items()}

    log(f"    ✓ Ensemble weights: {best}  RMSE={study.best_value:.0f}")
    return best


# ════════════════════════════════════════════════════════════
#  HELPER: lưu / load model
# ════════════════════════════════════════════════════════════

def save_model(model, name: str):
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    path = os.path.join(config.MODELS_DIR, f"{name}.keras")
    model.save(path)
    return path


def load_model(name: str):
    import tensorflow as tf
    return tf.keras.models.load_model(
        os.path.join(config.MODELS_DIR, f"{name}.keras")
    )
