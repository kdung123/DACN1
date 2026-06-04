# ╔══════════════════════════════════════════════════════════════╗
# ║                        models.py                            ║
# ║           Tất cả mô hình dự báo giá cổ phiếu               ║
# ╠══════════════════════════════════════════════════════════════╣
# ║  Thứ tự train trong pipeline:                              ║
# ║    1. Ridge(Optuna)  — linear, L2 regularization           ║
# ║    2. RF(Optuna)     — non-linear, tree ensemble            ║
# ║    3. BiGRU          — deep learning, chuỗi thời gian       ║
# ║    4. Ensemble       — kết hợp 3 model trên                ║
# ╠══════════════════════════════════════════════════════════════╣
# ║  Chống overfitting:                                        ║
# ║    Ridge   : L2 penalty (alpha tìm bằng Optuna)            ║
# ║    RF      : max_depth≤15, min_samples_leaf≥2              ║
# ║    BiGRU   : Dropout(0.3) + recurrent_dropout(0.1) + L2   ║
# ║              + EarlyStopping + ReduceLROnPlateau           ║
# ║  Chống data leak trong Ensemble:                           ║
# ║    Weights tìm trên VAL → apply lên TEST                   ║
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
#  MODEL 1 — RIDGE REGRESSION + OPTUNA
#
#  Tại sao Ridge thay vì LinearRegression?
#    LinearRegression : loss = MSE
#    Ridge            : loss = MSE + alpha × Σ(wᵢ²)
#    L2 penalty buộc weights nhỏ → ít nhạy cảm với noise
#    → phù hợp khi features tương quan cao (SMA, EMA, Lag)
#
#  Tại sao Optuna tìm alpha?
#    alpha quá nhỏ → gần như LinearRegression, dễ overfit
#    alpha quá lớn → weights → 0, model không học được gì
#    → Optuna tìm điểm tối ưu trên val_mse
# ════════════════════════════════════════════════════════════════

def train_ridge_optuna(X_train, y_train, X_val, y_val,
                       n_trials: int = config.RIDGE_TRIALS,
                       send_log=None):
    """
    Huấn luyện Ridge Regression với Optuna tìm alpha tối ưu.

    Quy trình
    ─────────
    1. Optuna thử n_trials giá trị alpha trên log-scale [1e-3, 100]
       → tìm alpha cho val_mse thấp nhất
    2. Fit lại model trên train+val với alpha tốt nhất
       → tận dụng toàn bộ dữ liệu có nhãn trước test

    Parameters
    ----------
    n_trials : int   Số lần thử (30 đủ cho không gian 1 chiều)

    Returns
    -------
    (model, best_alpha)
    """
    from sklearn.linear_model import Ridge
    from sklearn.metrics import mean_squared_error

    log = send_log or print
    log(f"    Ridge Optuna: {n_trials} trials  (alpha ∈ [1e-3, 100] log-scale)")

    # ── Optuna tìm alpha tốt nhất ────────────────────────────
    def objective(trial):
        alpha = trial.suggest_float("alpha", 1e-3, 10, log=True)
        model = Ridge(alpha=alpha).fit(X_train, y_train)
        return float(mean_squared_error(y_val, model.predict(X_val)))

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_alpha = study.best_params["alpha"]
    log(f"    ✓ best alpha = {best_alpha:.4f}  |  val_mse = {study.best_value:.6f}")

    # ── Fit lại trên train + val với alpha tốt nhất ──────────
    # Sau khi chọn alpha qua val, fit lại trên train+val
    # để model học từ nhiều dữ liệu hơn trước khi predict test
    final = Ridge(alpha=best_alpha).fit(
        np.concatenate([X_train, X_val]),
        np.concatenate([y_train, y_val]),
    )
    return final, best_alpha


# ════════════════════════════════════════════════════════════════
#  MODEL 2 — RANDOM FOREST + OPTUNA
#
#  Tại sao RF tốt hơn Decision Tree đơn?
#    RF = trung bình nhiều cây — mỗi cây train trên bootstrap sample
#    → variance thấp hơn, ít overfit hơn cây đơn
#
#  Chống overfitting:
#    max_depth ≤ 15    : giới hạn độ sâu cây
#    min_samples_leaf ≥ 2 : lá phải có ít nhất 2 mẫu
#    max_features      : mỗi split chỉ xem một phần features
#                        → tăng đa dạng giữa các cây
# ════════════════════════════════════════════════════════════════

def train_rf_optuna(X_train, y_train, X_val, y_val,
                    n_trials: int = config.RF_TRIALS,
                    send_log=None):
    """
    Huấn luyện Random Forest với Optuna tìm 4 hyperparameters.

    Không gian tìm kiếm
    ───────────────────
    n_estimators     : 50–300 (bước 50)  — số cây
    max_depth        : 5–15              — độ sâu tối đa (≤15 tránh overfit)
    min_samples_leaf : 2–10             — tối thiểu 2 mẫu/lá
    max_features     : sqrt|log2|0.5    — features mỗi split

    Returns
    -------
    (model, best_params)
    """
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_squared_error

    log = send_log or print
    log(f"    RF Optuna: {n_trials} trials  (4 hyperparameters)")

    # ── Optuna tìm hyperparameters ───────────────────────────
    def objective(trial):
        params = {
            "n_estimators":     trial.suggest_int("n_estimators", 50, 300, step=50),
            "max_depth":        trial.suggest_int("max_depth", 5, 15),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 2, 10),
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

    # ── Fit lại trên train + val ──────────────────────────────
    final = RandomForestRegressor(
        **bp, random_state=config.RANDOM_SEED, n_jobs=-1
    ).fit(
        np.concatenate([X_train, X_val]),
        np.concatenate([y_train, y_val]),
    )
    return final, bp


# ════════════════════════════════════════════════════════════════
#  MODEL 3 — BIDIRECTIONAL GRU
#
#  Tại sao BiGRU thay vì GRU đơn hướng?
#    GRU đơn hướng: chỉ đọc chuỗi theo chiều thuận (T-N → T-1)
#    BiGRU: đọc cả chiều thuận lẫn chiều nghịch
#    → học được pattern "chuẩn bị cho đỉnh/đáy" tốt hơn
#
#  Kiến trúc
#  ─────────
#    Input(seq_len, n_features)
#      → Bidirectional(GRU(64, return_seq=True))
#      → Dropout(0.3)
#      → GRU(32, return_seq=False)
#      → Dropout(0.3)
#      → Dense(1)
#
#  Chống overfitting (4 lớp bảo vệ)
#  ──────────────────────────────────
#    1. Dropout(0.3)          — tắt 30% neuron mỗi bước train
#    2. recurrent_dropout(0.1) — tắt 10% hidden state giữa timesteps
#    3. L2(1e-4)              — penalize weights lớn
#    4. EarlyStopping(p=10)   — dừng khi val_loss không giảm
#    5. ReduceLROnPlateau(p=5) — giảm LR trước khi EarlyStopping dừng
# ════════════════════════════════════════════════════════════════

def build_bigru(input_shape: tuple,
                units: tuple = (64, 32),
                dropout: float = config.DROPOUT_RATE):
    """
    Xây dựng kiến trúc BiGRU (chưa train).

    Parameters
    ----------
    input_shape : (seq_len, n_features)
    units       : số unit GRU mỗi tầng
    dropout     : tỷ lệ dropout input

    Returns
    -------
    tf.keras.Model
    """
    import tensorflow as tf

    tf.random.set_seed(config.RANDOM_SEED)
    reg = tf.keras.regularizers.l2(1e-4)   # L2 regularization

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),

        # Tầng 1: BiGRU — đọc sequence theo 2 chiều
        # return_sequences=True → trả về output tại mỗi timestep
        # recurrent_dropout → dropout trên hidden state h(t-1)
        tf.keras.layers.Bidirectional(
            tf.keras.layers.GRU(units[0],
                                return_sequences=True,
                                kernel_regularizer=reg,
                                recurrent_dropout=0.1)
        ),
        tf.keras.layers.Dropout(dropout),

        # Tầng 2: GRU — tổng hợp toàn bộ sequence thành 1 vector
        # return_sequences=False → chỉ trả về output của timestep cuối
        tf.keras.layers.GRU(units[1],
                            return_sequences=False,
                            kernel_regularizer=reg,
                            recurrent_dropout=0.1),
        tf.keras.layers.Dropout(dropout),

        # Output: dự báo 1 giá trị (Close ngày T)
        tf.keras.layers.Dense(1),

    ], name="BiGRU")

    model.compile(optimizer="adam", loss="mse")
    return model


def train_bigru(Xs_tr, ys_tr, Xs_vl, ys_vl,
                epochs: int = None,
                send_log=None,
                send_progress=None):
    """
    Huấn luyện BiGRU với EarlyStopping + ReduceLROnPlateau.

    Callbacks
    ─────────
    EarlyStopping(patience=10)
      → dừng nếu val_loss không giảm sau 10 epoch liên tiếp
      → restore_best_weights: khôi phục weights tốt nhất

    ReduceLROnPlateau(patience=5, factor=0.5)
      → giảm LR ÷ 2 nếu val_loss không giảm sau 5 epoch
      → kích hoạt TRƯỚC EarlyStopping để thoát plateau

    Returns
    -------
    (model, history)
      history.history["loss"]     — train loss mỗi epoch
      history.history["val_loss"] — val loss mỗi epoch
    """
    import tensorflow as tf

    epochs   = epochs or config.DL_EPOCHS
    log      = send_log      or print
    progress = send_progress or (lambda p: None)

    # ── Callback: log từng epoch ──────────────────────────────
    class EpochLogger(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            # Cập nhật progress bar trên UI
            progress(min(int((epoch + 1) / epochs * 90) + 5, 95))
            lr = float(self.model.optimizer.learning_rate)
            log(f"      Epoch {epoch+1:>3}/{epochs}  "
                f"train={logs.get('loss', 0):.5f}  "
                f"val={logs.get('val_loss', 0):.5f}  "
                f"lr={lr:.1e}")

    model = build_bigru((Xs_tr.shape[1], Xs_tr.shape[2]))
    log(f"    BiGRU: {model.count_params():,} tham số  "
        f"(input: {Xs_tr.shape[1]} steps × {Xs_tr.shape[2]} features)")

    history = model.fit(
        Xs_tr, ys_tr,
        validation_data = (Xs_vl, ys_vl),
        epochs          = epochs,
        batch_size      = config.DL_BATCH_SIZE,
        verbose         = 0,
        callbacks       = [
            # Giảm LR trước khi dừng — thoát local minimum
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5,
                patience=5, min_lr=1e-6, verbose=0
            ),
            # Dừng khi val_loss không cải thiện
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=config.DL_PATIENCE,
                restore_best_weights=True, verbose=0
            ),
            EpochLogger(),
        ],
    )

    stopped = len(history.history["loss"])
    best_ep = int(np.argmin(history.history["val_loss"])) + 1
    log(f"    ✓ BiGRU  dừng ep {stopped}/{epochs}  "
        f"|  best val ep {best_ep}  "
        f"|  best val_loss = {min(history.history['val_loss']):.5f}")
    return model, history


# ════════════════════════════════════════════════════════════════
#  MODEL 4 — ENSEMBLE (Optuna tìm trọng số)
#
#  Ý tưởng:
#    Mỗi model giỏi ở vùng khác nhau của dữ liệu
#    → kết hợp theo trọng số tối ưu → RMSE thấp hơn model đơn lẻ
#
#  Chống data leak:
#    Weights tìm trên VAL (không dùng test)
#    → apply weights lên TEST để đánh giá
#    → test không tham gia vào bất kỳ bước tune nào
#
#  find_optimal_ensemble_weights() — gọi trước ensemble_predict()
#  ensemble_predict()              — gọi sau khi có weights
# ════════════════════════════════════════════════════════════════

def find_optimal_ensemble_weights(predictions: dict,
                                   y_true: np.ndarray,
                                   n_trials: int = config.ENSEMBLE_TRIALS,
                                   send_log=None) -> dict:
    """
    Optuna tìm bộ trọng số minimize RMSE trên tập VAL.

    Không gian tìm kiếm: N chiều liên tục [0, 1]
    Sau mỗi trial: normalize tổng trọng số = 1

    Parameters
    ----------
    predictions : dict         {model_name: np.ndarray}  pred trên VAL
    y_true      : np.ndarray   giá thực tế trên VAL
    n_trials    : int          số lần thử Optuna

    Returns
    -------
    dict  {model_name: weight}  tổng = 1.0
    """
    log   = send_log or print
    names = list(predictions.keys())
    arrs  = [np.array(predictions[n]) for n in names]

    # Căn chỉnh độ dài (BiGRU ngắn hơn seq_len phiên)
    n    = min(len(a) for a in arrs + [y_true])
    arrs = [a[-n:] for a in arrs]
    y    = y_true[-n:]

    def objective(trial):
        # Mỗi trial thử một bộ trọng số ngẫu nhiên
        w = np.array([trial.suggest_float(name, 0.0, 1.0) for name in names])
        if w.sum() < 1e-6:
            return 1e9
        w   /= w.sum()                              # normalize tổng = 1
        pred = sum(wi * a for wi, a in zip(w, arrs))
        return float(np.sqrt(np.mean((y - pred) ** 2)))  # minimize RMSE

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    # Normalize weights tốt nhất
    raw   = {n: study.best_params[n] for n in names}
    total = sum(raw.values())
    best  = {n: round(v / total, 3) for n, v in raw.items()}

    log(f"    ✓ Ensemble weights: {best}")
    log(f"       RMSE trên VAL = {study.best_value:.0f}")
    return best


def ensemble_predict(predictions: dict,
                     weights: dict = None) -> np.ndarray:
    """
    Kết hợp predictions theo trọng số.
    Tự động căn chỉnh độ dài về min_len.

    Parameters
    ----------
    predictions : dict   {model_name: np.ndarray}
    weights     : dict   {model_name: float}  None = trọng số bằng nhau

    Returns
    -------
    np.ndarray  round đến hàng đơn vị (float, không phải int)
    """
    names   = list(predictions.keys())
    arrs    = [np.array(predictions[n]) for n in names]
    min_len = min(len(a) for a in arrs)
    arrs    = [a[-min_len:] for a in arrs]

    if weights is None:
        # Trọng số bằng nhau nếu không cung cấp
        w = np.ones(len(arrs)) / len(arrs)
    else:
        w_raw = np.array([weights.get(n, 1.0) for n in names])
        w     = w_raw / w_raw.sum()

    pred = sum(wi * a for wi, a in zip(w, arrs))
    return np.round(pred, 0)   # round đến hàng đơn vị, giữ kiểu float


# ════════════════════════════════════════════════════════════════
#  HELPER — Lưu / Load model Keras
# ════════════════════════════════════════════════════════════════

def save_model(model, name: str) -> str:
    """Lưu Keras model vào thư mục models/ với định dạng .keras."""
    import os
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    path = os.path.join(config.MODELS_DIR, f"{name}.keras")
    model.save(path)
    return path


def load_model(name: str):
    """Load Keras model đã lưu từ thư mục models/."""
    import tensorflow as tf
    path = os.path.join(config.MODELS_DIR, f"{name}.keras")
    return tf.keras.models.load_model(path)
