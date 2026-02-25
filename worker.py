"""
OEnergy GPU Serverless Worker v1.0
RunPod Serverless handler for LSTM foundation training and transfer learning.

Operations:
  health          → GPU/TF status
  train_foundation → Train shared LSTM foundation model on windowed multi-code data
  transfer        → Transfer-learn per-code LSTM from foundation (WINDOWING BUG FIXED)
  transfer_batch  → Process list of transfer requests sequentially

Architecture: TensorFlow-GPU (same framework as TF.js → zero model conversion)
Model storage: S3 bucket (TF.js format: model.json + binary weights)
"""

import os
import time
import tempfile
import traceback

import boto3
import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import tensorflow as tf
import tensorflowjs as tfjs
import runpod

LOG = "[OEnergyGPUWorker]"

S3_BUCKET = os.environ.get("AWS_S3_BUCKET_NAME", "oenergyreseller")
S3_MODEL_PREFIX = "lstm-gpu"

_s3_client = None


def get_s3():
    global _s3_client
    if _s3_client is None:
        _s3_client = boto3.client(
            "s3",
            aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
            region_name=os.environ.get("AWS_REGION", "eu-south-1"),
        )
    return _s3_client


# ─────────────────────────────────────────────────────────────────────────────
# MODEL ARCHITECTURE  (identical to TF.js CPU model in lstm-foundation-engine.ts)
# ─────────────────────────────────────────────────────────────────────────────

def build_lstm_model(
    commodity: str,
    window_size: int,
    n_features: int,
    config: dict,
) -> tf.keras.Model:
    """
    LSTM architecture matching server/lstm-foundation-engine.ts buildLSTMModel():
      EE:  Input[30,146] → LSTM(128,seq) → Drop(0.2) → LSTM(64) → Drop(0.2) → Dense(64,relu) → Dense(96,sigmoid)
      Gas: Input[90,40]  → LSTM(64, seq) → Drop(0.2) → LSTM(32) → Drop(0.2) → Dense(32,relu) → Dense(1, sigmoid)
    """
    is_gas = commodity == "gas"
    l1 = config.get("lstmUnitsGasL1" if is_gas else "lstmUnitsL1", 64 if is_gas else 128)
    l2 = config.get("lstmUnitsGasL2" if is_gas else "lstmUnitsL2", 32 if is_gas else 64)
    dropout = config.get("dropoutRate", 0.2)
    output_units = 1 if is_gas else 96
    lr = config.get("learningRate", 0.001)

    inp = tf.keras.Input(shape=(window_size, n_features))
    x = tf.keras.layers.LSTM(
        l1,
        return_sequences=True,
        kernel_regularizer=tf.keras.regularizers.L2(0.001),
    )(inp)
    x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.LSTM(l2, return_sequences=False)(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Dense(l2, activation="relu")(x)
    out = tf.keras.layers.Dense(output_units, activation="sigmoid")(x)

    model = tf.keras.Model(inputs=inp, outputs=out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="mean_squared_error",
        metrics=["mae"],
    )
    return model


# ─────────────────────────────────────────────────────────────────────────────
# WINDOWING  — FIX for original PyTorch "too many indices" pre-windowing bug
# ─────────────────────────────────────────────────────────────────────────────

def create_windows(
    features: np.ndarray,
    targets: np.ndarray,
    window_size: int,
):
    """
    Sliding window: 2D [T, F] → 3D [N, W, F].

    Original bug: shape[2] (IndexError on 2D array), fixed to shape[1] / shape[-1].
    """
    if features.ndim != 2:
        raise ValueError(f"features must be 2D [T, F], got shape {features.shape}")

    n_samples = len(features) - window_size
    if n_samples <= 0:
        return None, None

    n_features = features.shape[1]  # shape[-1] — NOT shape[2] (the original bug)

    X = np.empty((n_samples, window_size, n_features), dtype=np.float32)
    for i in range(n_samples):
        X[i] = features[i : i + window_size]

    y = targets[window_size:]
    return X, y


# ─────────────────────────────────────────────────────────────────────────────
# S3 HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def upload_model_to_s3(model: tf.keras.Model, s3_prefix: str, tmp_dir: str) -> str:
    """Export Keras model to TF.js format and upload to S3. Returns s3_prefix."""
    model_dir = os.path.join(tmp_dir, "tfjs_model")
    os.makedirs(model_dir, exist_ok=True)

    tfjs.converters.save_keras_model(model, model_dir)

    s3 = get_s3()
    for fname in os.listdir(model_dir):
        local_path = os.path.join(model_dir, fname)
        s3_key = f"{s3_prefix}/{fname}"
        s3.upload_file(local_path, S3_BUCKET, s3_key)
        print(f"{LOG} Uploaded s3://{S3_BUCKET}/{s3_key}")

    return s3_prefix


def download_model_from_s3(s3_prefix: str, tmp_dir: str) -> tf.keras.Model:
    """Download TF.js model files from S3 and load as Keras model."""
    model_dir = os.path.join(tmp_dir, "model")
    os.makedirs(model_dir, exist_ok=True)

    s3 = get_s3()
    paginator = s3.get_paginator("list_objects_v2")
    found_any = False
    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=s3_prefix + "/"):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            fname = os.path.basename(key)
            local_path = os.path.join(model_dir, fname)
            s3.download_file(S3_BUCKET, key, local_path)
            found_any = True

    if not found_any:
        raise FileNotFoundError(f"No model files at s3://{S3_BUCKET}/{s3_prefix}/")

    return tfjs.converters.load_keras_model(os.path.join(model_dir, "model.json"))


# ─────────────────────────────────────────────────────────────────────────────
# OPERATION HANDLERS
# ─────────────────────────────────────────────────────────────────────────────

def handle_health(_job_input: dict) -> dict:
    gpus = tf.config.list_physical_devices("GPU")
    return {
        "status": "healthy",
        "tensorflow_version": tf.__version__,
        "gpu_available": len(gpus) > 0,
        "gpu_count": len(gpus),
        "gpu_devices": [g.name for g in gpus],
        "s3_bucket": S3_BUCKET,
        "s3_model_prefix": S3_MODEL_PREFIX,
        "worker": "OEnergy GPU Serverless Worker v1.0",
    }


def handle_train_foundation(job_input: dict) -> dict:
    """
    Train LSTM foundation model from pre-windowed multi-code data.

    Input:
      commodity:     'electricity' | 'gas'
      training_data: [{ code, features[N][W][F], targets[N][outputs] }]
      config:        { lstmUnitsL1, lstmUnitsL2, dropoutRate, learningRate, batchSize, ... }
      epochs:        int (default 50)
    """
    t0 = time.time()
    commodity = job_input["commodity"]
    training_data = job_input["training_data"]
    config = job_input.get("config", {})
    epochs = job_input.get("epochs", 50)
    batch_size = config.get("batchSize", 32)

    print(f"{LOG} Foundation training: {commodity}, {len(training_data)} codes, epochs={epochs}")

    all_X, all_y = [], []
    for item in training_data:
        X = np.array(item["features"], dtype=np.float32)
        y = np.array(item["targets"], dtype=np.float32)
        if X.ndim != 3:
            raise ValueError(f"Foundation features for {item.get('code')} must be 3D [N,W,F], got {X.shape}")
        all_X.append(X)
        all_y.append(y)

    if not all_X:
        return {"success": False, "error": "No training data provided"}

    X_train = np.concatenate(all_X, axis=0)
    y_train = np.concatenate(all_y, axis=0)
    window_size = int(X_train.shape[1])
    n_features = int(X_train.shape[2])

    print(f"{LOG} Training tensor: X={X_train.shape}, y={y_train.shape}")

    model = build_lstm_model(commodity, window_size, n_features, config)
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        verbose=0,
        callbacks=[tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)],
    )

    best_mae = float(min(history.history.get("val_mae", history.history["mae"])))
    best_loss = float(min(history.history.get("val_loss", history.history["loss"])))
    epochs_done = len(history.history["loss"])

    s3_prefix = f"{S3_MODEL_PREFIX}/foundation_{commodity}"
    with tempfile.TemporaryDirectory() as tmp:
        upload_model_to_s3(model, s3_prefix, tmp)

    elapsed = time.time() - t0
    print(f"{LOG} Foundation done: mae={best_mae:.4f}, time={elapsed:.1f}s")

    return {
        "success": True,
        "commodity": commodity,
        "model_id": f"foundation_{commodity}",
        "s3_prefix": s3_prefix,
        "samples_used": len(training_data),
        "total_windows": int(X_train.shape[0]),
        "epochs_completed": epochs_done,
        "best_loss": best_loss,
        "mae": best_mae,
        "training_time_seconds": elapsed,
    }


def handle_transfer(job_input: dict) -> dict:
    """
    Transfer learning for a single code.

    Input:
      code:               str
      commodity:          'electricity' | 'gas'
      features:           List[List[float]]  — shape [T][F] RAW 2D (WINDOWING DONE HERE, CORRECTLY)
      targets:            List[List[float]]  — shape [T][outputs]
      foundation_model_id: str (default 'foundation_{commodity}')
      config:             { learningRate, batchSize, ... }
      epochs:             int (default 30)
    """
    t0 = time.time()
    code = job_input["code"]
    commodity = job_input["commodity"]
    features_2d = job_input["features"]
    targets_2d = job_input["targets"]
    foundation_model_id = job_input.get("foundation_model_id", f"foundation_{commodity}")
    config = job_input.get("config", {})
    epochs = job_input.get("epochs", 30)
    batch_size = config.get("batchSize", 16)

    is_gas = commodity == "gas"
    window_size = 90 if is_gas else 30

    print(f"{LOG} Transfer {code} ({commodity}): {len(features_2d)} timesteps, epochs={epochs}")

    features_arr = np.array(features_2d, dtype=np.float32)
    targets_arr = np.array(targets_2d, dtype=np.float32)

    if features_arr.ndim != 2:
        return {
            "success": False,
            "code": code,
            "error": f"features must be 2D [T,F], got {features_arr.shape}",
        }

    n_features = features_arr.shape[1]

    X, y = create_windows(features_arr, targets_arr, window_size)
    if X is None or len(X) < 5:
        return {
            "success": False,
            "code": code,
            "error": f"Too few windows ({len(features_2d)} timesteps, need >{window_size}+5)",
        }

    print(f"{LOG} Transfer {code}: X={X.shape}, y={y.shape}")

    foundation_model = None
    foundation_s3_prefix = f"{S3_MODEL_PREFIX}/{foundation_model_id}"
    try:
        with tempfile.TemporaryDirectory() as tmp:
            foundation_model = download_model_from_s3(foundation_s3_prefix, tmp)
        print(f"{LOG} Loaded foundation model from s3://{S3_BUCKET}/{foundation_s3_prefix}")
    except Exception as e:
        print(f"{LOG} Foundation model not found ({e}), training from scratch")

    if foundation_model is not None:
        model = tf.keras.models.clone_model(foundation_model)
        model.set_weights(foundation_model.get_weights())
        lr = config.get("learningRate", 0.001) * 0.1
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
            loss="mean_squared_error",
            metrics=["mae"],
        )
        del foundation_model
    else:
        model = build_lstm_model(commodity, window_size, n_features, config)

    val_size = max(5, int(len(X) * 0.2))
    history = model.fit(
        X, y,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=min(0.2, val_size / len(X)),
        verbose=0,
        callbacks=[tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)],
    )

    val_mae_hist = history.history.get("val_mae", history.history["mae"])
    val_loss_hist = history.history.get("val_loss", history.history["loss"])
    epochs_done = len(history.history["loss"])

    y_pred = model.predict(X, verbose=0)
    mae = float(np.mean(np.abs(y - y_pred)))
    rmse = float(np.sqrt(np.mean((y - y_pred) ** 2)))
    epsilon = 1e-8
    mape = float(np.mean(np.abs((y - y_pred) / (np.abs(y) + epsilon))) * 100)

    s3_prefix = f"{S3_MODEL_PREFIX}/{code}_{commodity}"
    with tempfile.TemporaryDirectory() as tmp:
        upload_model_to_s3(model, s3_prefix, tmp)

    elapsed = time.time() - t0
    print(f"{LOG} Transfer {code} done: mae={mae:.4f}, rmse={rmse:.4f}, time={elapsed:.1f}s")

    return {
        "success": True,
        "code": code,
        "commodity": commodity,
        "s3_prefix": s3_prefix,
        "epochs_completed": epochs_done,
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
        "max_reliable_horizon": 1 if is_gas else 96,
        "training_time_seconds": elapsed,
        "used_foundation": foundation_model is not None,
    }


def handle_transfer_batch(job_input: dict) -> dict:
    """
    Process a list of transfer learning requests sequentially on the same GPU.

    Input:
      points: List of transfer request objects (same schema as handle_transfer)
    """
    t0 = time.time()
    points = job_input.get("points", [])

    print(f"{LOG} Batch transfer: {len(points)} codes")

    results = []
    success_count = 0
    error_count = 0

    for i, point in enumerate(points):
        try:
            print(f"{LOG} Batch {i+1}/{len(points)}: {point.get('code')}")
            result = handle_transfer(point)
            results.append(result)
            if result.get("success"):
                success_count += 1
            else:
                error_count += 1
        except Exception as e:
            error_count += 1
            results.append({
                "success": False,
                "code": point.get("code", "unknown"),
                "error": str(e),
                "traceback": traceback.format_exc(),
            })

    return {
        "results": results,
        "total": len(points),
        "success": success_count,
        "errors": error_count,
        "total_time_seconds": time.time() - t0,
    }


# ─────────────────────────────────────────────────────────────────────────────
# RUNPOD HANDLER
# ─────────────────────────────────────────────────────────────────────────────

def handler(job: dict) -> dict:
    """RunPod Serverless entry point."""
    job_input = job.get("input", {})
    operation = job_input.get("operation", "health")

    print(f"{LOG} Job {job.get('id', '?')}: operation={operation}")

    try:
        if operation == "health":
            return handle_health(job_input)
        elif operation == "train_foundation":
            return handle_train_foundation(job_input)
        elif operation == "transfer":
            return handle_transfer(job_input)
        elif operation == "transfer_batch":
            return handle_transfer_batch(job_input)
        else:
            return {"success": False, "error": f"Unknown operation: {operation}"}
    except Exception as e:
        tb = traceback.format_exc()
        print(f"{LOG} UNHANDLED ERROR in {operation}: {e}\n{tb}")
        return {"success": False, "error": str(e), "traceback": tb}


if __name__ == "__main__":
    print(f"{LOG} Starting RunPod Serverless worker...")
    gpus = tf.config.list_physical_devices("GPU")
    print(f"{LOG} TensorFlow {tf.__version__}, GPUs: {[g.name for g in gpus]}")
    runpod.serverless.start({"handler": handler})
