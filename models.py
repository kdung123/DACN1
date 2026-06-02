# ╔══════════════════════════════════════════════════════════════╗
# ║                        models.py                            ║
# ║           Tất cả mô hình dự báo giá cổ phiếu               ║
# ╠══════════════════════════════════════════════════════════════╣
# ║  1. Ridge Regression  — L2 regularization + Optuna alpha    ║
# ║  2. Random Forest     — Optuna tìm 4 hyperparameters        ║
# ║  3. Bidirectional GRU — Dropout 0.3 + L2 + ReduceLR        ║
# ║  4. Ensemble          — Optuna tìm trọng số tối ưu          ║
# ╚══════════════════════════════════════════════════════════════╝

import os
import warnings

import numpy as np
import optuna

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")

import config


# ════════════════════════════════════════════════════════════════
#  1. RIDGE REGRESSION + OPTUNA
# ════════════════════════════════════════════════════════════════

def train_ridge_optuna(X_train, y_train, X_val, y_val,
                       n_trials: int = config.RIDGE_TRIALS,
                       send_log=None):
    """
    Ridge Regression với L2 regularization.
    Optuna tự động tìm hệ số alpha tối ưu trên không gian log-scale.

    So với LinearRegression thuần:
      LinearRegression : loss = MSE
      Ridge            : loss = MSE + alpha × Σ(wᵢ²)
    → Ridge ít overfitting hơn, ổn định hơn khi features tương quan cao.

    Parameters
    ----------
    n_trials : int   Số lần thử Optuna (mặc định 30 — đủ cho 1 chiều)

    Returns
    -------
    (model, best_alpha)
      model      : Ridge đã fit lại trên train + val
      best_alpha : alpha tối ưu tìm được
    """
    from sklearn.linear_model import Ridge
    from sklearn.metrics import mean_squared_error

    log = send_log or print
    log(f"    Optuna Ridge: {n_trials} trials tìm alpha (log-scale 1e-3 → 100)...")

    def objective(trial):
        alpha = trial.suggest_float("alpha", 1e-3, 100, log=True)
        model = Ridge(alpha=alpha).fit(X_train, y_train)
        return float(mean_squared_error(y_val, model.predict(X_val)))

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_alpha = study.best_params["alpha"]
    log(f"    ✓ alpha = {best_alpha:.4f}  |  val_mse = {study.best_value:.6f}")

    # Fit lại trên train + val để tận dụng toàn bộ dữ liệu có nhãn
    final_model = Ridge(alpha=best_alpha).fit(
        np.concatenate([X_train, X_val]),
        np.concatenate([y_train, y_val]),
    )
    return final_model, best_alpha


# ════════════════════════════════════════════════════════════════
#  2. RANDOM FOREST + OPTUNA
# ════════════════════════════════════════════════════════════════

def train_rf_optuna(X_train, y_train, X_val, y_val,
                    n_trials: int = config.RF_TRIALS,
                    send_log=None):
    """
    Random Forest với Optuna tìm 4 hyperparameters cùng lúc:
      • n_estimators     : số cây (50 → 300)
      • max_depth        : độ sâu tối đa (5 → 25)
      • min_samples_leaf : số mẫu tối thiểu ở lá (1 → 10)
      • max_features     : số features mỗi split ("sqrt" / "log2" / 0.5)

    Parameters
    ----------
    n_trials : int   Số lần thử Optuna (mặc định 40 — phù hợp 4 chiều)

    Returns
    -------
    (model, best_params)
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
            "max_features":     trial.suggest_categorical("max_features",
                                                          ["sqrt", "log2", 0.5]),
        }
        model = RandomForestRegressor(
            **params, random_state=config.RANDOM_SEED, n_jobs=-1
        ).fit(X_train, y_train)
        return float(mean_squared_error(y_val, model.predict(X_val)))

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    bp = study.best_params
    log(f"    ✓ n_est={bp['n_estimators']}  depth={bp['max_depth']}  "
        f"leaf={bp['min_samples_leaf']}  feats={bp['max_features']}")
    log(f"       val_mse = {study.best_value:.4f}")

    final_model = RandomForestRegressor(
        **bp, random_state=config.RANDOM_SEED, n_jobs=-1
    ).fit(
        np.concatenate([X_train, X_val]),
        np.concatenate([y_train, y_val]),
    )
    return final_model, bp


# ════════════════════════════════════════════════════════════════
#  3. BIDIRECTIONAL GRU
# ════════════════════════════════════════════════════════════════

def build_bigru(input_shape: tuple,
                units: tuple = (64, 32),
                dropout: float = config.DROPOUT_RATE):
    """
    Kiến trúc Bidirectional GRU:

        Input(seq_len, n_features)
          → Bidirectional(GRU(64, return_sequences=True))  ← đọc 2 chiều
          → Dropout(0.3)
          → GRU(32, return_sequences=False)
          → Dropout(0.3)
          → Dense(1)

    Cải tiến so với GRU đơn hướng:
      • Bidirectional: học pattern theo cả chiều thuận và nghịch
      • Dropout 0.3 + L2(1e-4): giảm overfitting trên dữ liệu ngắn
      • ReduceLROnPlateau: tự động giảm learning rate khi plateau

    Parameters
    ----------
    input_shape : tuple   (seq_len, n_features)
    units       : tuple   Số unit GRU mỗi tầng
    dropout     : float   Tỷ lệ dropout

    Returns
    -------
    tf.keras.Model   (chưa train)
    """
    import tensorflow as tf

    tf.random.set_seed(config.RANDOM_SEED)
    reg = tf.keras.regularizers.l2(1e-4)

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),

        tf.keras.layers.Bidirectional(
            tf.keras.layers.GRU(units[0], return_sequences=True,
                                kernel_regularizer=reg)
        ),
        tf.keras.layers.Dropout(dropout),

        tf.keras.layers.GRU(units[1], return_sequences=False,
                            kernel_regularizer=reg),
        tf.keras.layers.Dropout(dropout),

        tf.keras.layers.Dense(1),
    ], name="BiGRU")

    model.compile(optimizer="adam", loss="mse")
    return model


def train_bigru(Xs_tr, ys_tr, Xs_vl, ys_vl,
                epochs: int = None,
                send_log=None,
                send_progress=None):
    """
    Huấn luyện BiGRU với:
      • EarlyStopping(patience=10) + restore_best_weights
      • ReduceLROnPlateau(patience=5, factor=0.5, min_lr=1e-6)

    Returns
    -------
    (model, history)
      model   : BiGRU đã train
      history : Keras History object (dùng để vẽ loss curve)
    """
    import tensorflow as tf

    epochs   = epochs   or config.DL_EPOCHS
    log      = send_log      or print
    progress = send_progress or (lambda p: None)

    # ── Callback log từng epoch ───────────────────────────────
    class EpochLogger(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            progress(min(int((epoch + 1) / epochs * 90) + 5, 95))
            lr = float(self.model.optimizer.learning_rate)
            log(f"      Epoch {epoch+1:>3}/{epochs}  "
                f"train={logs.get('loss', 0):.5f}  "
                f"val={logs.get('val_loss', 0):.5f}  "
                f"lr={lr:.1e}")

    model = build_bigru((Xs_tr.shape[1], Xs_tr.shape[2]))
    log(f"    BiGRU: {model.count_params():,} tham số")

    history = model.fit(
        Xs_tr, ys_tr,
        validation_data = (Xs_vl, ys_vl),
        epochs          = epochs,
        batch_size      = config.DL_BATCH_SIZE,
        verbose         = 0,
        callbacks       = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=config.DL_PATIENCE,
                restore_best_weights=True, verbose=0
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5,
                patience=5, min_lr=1e-6, verbose=0
            ),
            EpochLogger(),
        ],
    )

    stopped = len(history.history["loss"])
    best_ep = int(np.argmin(history.history["val_loss"])) + 1
    log(f"    ✓ BiGRU  dừng ep {stopped}/{epochs}  |  best val ep {best_ep}")
    return model, history


# ════════════════════════════════════════════════════════════════
#  4. ENSEMBLE (Optuna tìm trọng số)
# ════════════════════════════════════════════════════════════════

def ensemble_predict(predictions: dict, weights: dict = None) -> np.ndarray:
    """
    Kết hợp dự báo nhiều model theo trọng số có trọng số.
    Tự động căn chỉnh độ dài về min_len (an toàn khi các pred khác nhau).

    Parameters
    ----------
    predictions : dict   {model_name: np.ndarray}
    weights     : dict   {model_name: float} — None = trọng số bằng nhau

    Returns
    -------
    np.ndarray   Dự báo ensemble (float, round đến hàng đơn vị)
    """
    names   = list(predictions.keys())
    arrs    = [np.array(predictions[n]) for n in names]
    min_len = min(len(a) for a in arrs)
    arrs    = [a[-min_len:] for a in arrs]

    if weights is None:
        w = np.ones(len(arrs)) / len(arrs)
    else:
        w_raw = np.array([weights.get(n, 1.0) for n in names])
        w     = w_raw / w_raw.sum()

    pred = sum(wi * a for wi, a in zip(w, arrs))
    return np.round(pred, 0)   # float để giữ precision, round đến đơn vị


def find_optimal_ensemble_weights(predictions: dict,
                                   y_true: np.ndarray,
                                   n_trials: int = config.ENSEMBLE_TRIALS,
                                   send_log=None) -> dict:
    """
    Optuna tìm bộ trọng số tối ưu cho Ensemble bằng cách minimize RMSE.

    Với N model → không gian N chiều liên tục [0, 1].
    Sau mỗi trial, Optuna normalize tổng trọng số = 1.

    Parameters
    ----------
    n_trials : int   Số lần thử (mặc định 100 — phù hợp ≥ 3 model)

    Returns
    -------
    dict   {model_name: weight}  tổng = 1.0
    """
    log   = send_log or print
    names = list(predictions.keys())
    arrs  = [np.array(predictions[n]) for n in names]
    n     = min(len(a) for a in arrs + [y_true])
    arrs  = [a[-n:] for a in arrs]
    y     = y_true[-n:]

    def objective(trial):
        w = np.array([trial.suggest_float(name, 0.0, 1.0) for name in names])
        if w.sum() < 1e-6:
            return 1e9
        w   /= w.sum()
        pred = sum(wi * a for wi, a in zip(w, arrs))
        return float(np.sqrt(np.mean((y - pred) ** 2)))

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    raw   = {n: study.best_params[n] for n in names}
    total = sum(raw.values())
    best  = {n: round(v / total, 3) for n, v in raw.items()}

    log(f"    ✓ Weights: {best}")
    log(f"       RMSE ensemble = {study.best_value:.0f}")
    return best


# ════════════════════════════════════════════════════════════════
#  HELPER: Lưu / Load model Keras
# ════════════════════════════════════════════════════════════════

def save_model(model, name: str) -> str:
    """Lưu Keras model vào thư mục models/."""
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    path = os.path.join(config.MODELS_DIR, f"{name}.keras")
    model.save(path)
    return path


def load_model(name: str):
    """Load Keras model từ thư mục models/."""
    import tensorflow as tf
    path = os.path.join(config.MODELS_DIR, f"{name}.keras")
    return tf.keras.models.load_model(path)
