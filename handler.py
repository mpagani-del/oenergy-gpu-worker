import runpod
import time
import base64
import zlib
import json
import math
import numpy as np
import os
import urllib.request
import urllib.error

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.keras.layers import (
    Input, LSTM, Dense, Dropout, BatchNormalization,
    Reshape, Softmax, Multiply
)
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2

LOG = '[GPU-Worker]'
weight_cache = {}

ALLOWED_URL_PREFIX = 'https://storage.googleapis.com/'
MAX_DOWNLOAD_BYTES = 500 * 1024 * 1024
DOWNLOAD_TIMEOUT_S = 60


def decode_base64_to_float32(b64_string, is_compressed=False):
    buf = base64.b64decode(b64_string)
    if is_compressed:
        try:
            buf = zlib.decompress(buf)
        except zlib.error:
            pass
    return np.frombuffer(buf, dtype=np.float32).copy()


def decode_base64_weights(weights_b64, is_compressed=False):
    result = []
    for w in weights_b64:
        data = decode_base64_to_float32(w['b64'], is_compressed)
        result.append(np.reshape(data, w['shape']))
    return result


def get_foundation_weights(input_data):
    foundation_b64 = input_data.get('foundationWeightsB64')
    cache_key = input_data.get('weightCacheKey')
    foundation_raw = input_data.get('foundationWeights')
    is_compressed = input_data.get('compressed', False)

    if foundation_b64 and len(foundation_b64) > 0:
        decoded = decode_base64_weights(foundation_b64, is_compressed)
        if cache_key:
            weight_cache[cache_key] = foundation_b64
            weight_cache[f"{cache_key}_compressed"] = is_compressed
            print(f"{LOG} Foundation weights cached as '{cache_key}' ({len(foundation_b64)} tensors, compressed={is_compressed})")
        return decoded

    if cache_key and cache_key in weight_cache:
        cached_compressed = weight_cache.get(f"{cache_key}_compressed", False)
        print(f"{LOG} Using cached weights '{cache_key}' (compressed={cached_compressed})")
        return decode_base64_weights(weight_cache[cache_key], cached_compressed)

    if foundation_raw and len(foundation_raw) > 0:
        return [np.array(w['data'], dtype=np.float32).reshape(w['shape']) for w in foundation_raw]

    return None


def apply_attention(sequence, window_size, attention_units):
    attn_hidden = Dense(attention_units, activation='tanh')(sequence)
    attn_scores = Dense(1)(attn_hidden)
    attn_flat = Reshape((window_size,))(attn_scores)
    attn_weights_flat = Softmax()(attn_flat)
    attn_weights = Reshape((window_size, 1))(attn_weights_flat)
    return Multiply()([sequence, attn_weights])


def build_model(commodity, feature_count, window_size, config):
    is_ee = commodity == 'electricity'
    l1 = config.get('lstmUnitsL1') or (256 if is_ee else 128)
    l2_units = config.get('lstmUnitsL2') or (128 if is_ee else 64)
    l3 = config.get('lstmUnitsL3') or (64 if is_ee else 32)
    attn_units = config.get('attentionUnits') or 32
    output_units = 96 if is_ee else 1

    inp = Input(shape=(window_size, feature_count))
    x = inp

    x = LSTM(l1, return_sequences=True,
             kernel_regularizer=l2(0.001),
             recurrent_initializer='glorot_uniform')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    x = LSTM(l2_units, return_sequences=True,
             kernel_regularizer=l2(0.001),
             recurrent_initializer='glorot_uniform')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)

    x = apply_attention(x, window_size, attn_units)

    x = LSTM(l3, return_sequences=False,
             kernel_regularizer=l2(0.001),
             recurrent_initializer='glorot_uniform')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.1)(x)

    x = Dense(l2_units, activation='relu')(x)
    out = Dense(output_units, activation='sigmoid')(x)

    return Model(inputs=inp, outputs=out)


def serialize_weights(model):
    result = []
    for w in model.get_weights():
        result.append({
            'shape': list(w.shape),
            'data': w.flatten().tolist()
        })
    return result


def serialize_weights_compressed(model):
    result = []
    for w in model.get_weights():
        raw_bytes = w.astype(np.float32).tobytes()
        compressed = zlib.compress(raw_bytes)
        b64 = base64.b64encode(compressed).decode('ascii')
        result.append({
            'shape': list(w.shape),
            'b64': b64
        })
    return result


def safe_download(url, max_bytes=MAX_DOWNLOAD_BYTES, timeout=DOWNLOAD_TIMEOUT_S):
    req = urllib.request.Request(url)
    resp = urllib.request.urlopen(req, timeout=timeout)
    content_length = resp.headers.get('Content-Length')
    if content_length and int(content_length) > max_bytes:
        resp.close()
        raise ValueError(f"Download too large: {content_length} bytes exceeds {max_bytes} limit")
    chunks = []
    total = 0
    while True:
        chunk = resp.read(1024 * 1024)
        if not chunk:
            break
        total += len(chunk)
        if total > max_bytes:
            resp.close()
            raise ValueError(f"Download exceeded {max_bytes} byte limit during read")
        chunks.append(chunk)
    resp.close()
    return b''.join(chunks)


def mask_url(url):
    idx = url.find('?')
    if idx >= 0:
        return url[:idx] + '?<redacted>'
    return url


def derive_blob_url(manifest_url, filename):
    idx = manifest_url.rfind('/')
    base = manifest_url[:idx]
    query_idx = manifest_url.find('?')
    query = manifest_url[query_idx:] if query_idx >= 0 else ''
    return f"{base}/{filename}{query}"


def handler_foundation(input_data, config, start_time):
    code = input_data.get('code')
    commodity = input_data.get('commodity')

    data_url = input_data.get('dataUrl')
    if not data_url:
        return {'success': False, 'error': 'Foundation mode requires dataUrl'}

    if not data_url.startswith(ALLOWED_URL_PREFIX):
        return {'success': False, 'error': f'dataUrl must start with {ALLOWED_URL_PREFIX}'}

    print(f"{LOG} Foundation mode for {code} ({commodity}), downloading manifest from Object Storage...")

    try:
        manifest_raw = safe_download(data_url)
        manifest = json.loads(manifest_raw)
    except Exception as e:
        return {'success': False, 'error': f'Failed to download manifest: {str(e)}'}

    x_train_shape = manifest.get('xTrainShape')
    y_train_shape = manifest.get('yTrainShape')
    x_val_shape = manifest.get('xValShape')
    y_val_shape = manifest.get('yValShape')

    if not all([x_train_shape, y_train_shape, x_val_shape, y_val_shape]):
        return {'success': False, 'error': 'Manifest missing required shape fields'}

    blob_urls = manifest.get('blobUrls', {})

    blob_files = [
        ('x_train.bin', x_train_shape),
        ('y_train.bin', y_train_shape),
        ('x_val.bin', x_val_shape),
        ('y_val.bin', y_val_shape),
    ]

    arrays = {}
    for filename, shape in blob_files:
        blob_url = blob_urls.get(filename) or derive_blob_url(data_url, filename)
        if not blob_url.startswith(ALLOWED_URL_PREFIX):
            return {'success': False, 'error': f'Blob URL for {filename} has invalid prefix'}
        print(f"{LOG} Downloading {filename} (expected shape {shape})...")
        try:
            raw = safe_download(blob_url)
        except Exception as e:
            return {'success': False, 'error': f'Failed to download {filename}: {str(e)}'}

        expected_bytes = int(np.prod(shape)) * 4
        if len(raw) != expected_bytes:
            return {
                'success': False,
                'error': f'data integrity check failed: {filename} has {len(raw)} bytes, expected {expected_bytes}'
            }

        arr = np.frombuffer(raw, dtype=np.float32).reshape(shape)
        arrays[filename] = arr
        del raw
        print(f"{LOG} {filename} loaded: shape={list(arr.shape)}")

    x_train = arrays['x_train.bin']
    y_train = arrays['y_train.bin']
    x_val = arrays['x_val.bin']
    y_val = arrays['y_val.bin']
    del arrays

    window_size = x_train.shape[1]
    feature_count = x_train.shape[2]
    epochs = config.get('epochs', 30)
    learning_rate = config.get('learningRate', 0.0005)
    batch_size = min(config.get('batchSize', 32), max(16, len(x_train) // 8))

    print(f"{LOG} Foundation training {code} ({commodity}), x_train={list(x_train.shape)}, x_val={list(x_val.shape)}, epochs={epochs}, batch={batch_size}")

    try:
        model = build_model(commodity, feature_count, window_size, config)

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss='mean_squared_error',
            metrics=['mae']
        )

        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=7,
            restore_best_weights=True,
            verbose=0
        )

        history = model.fit(
            x_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(x_val, y_val),
            verbose=0,
            callbacks=[early_stop]
        )

        actual_epochs = len(history.history['loss'])
        val_loss_list = history.history.get('val_loss', [])
        best_val_loss = float(min(val_loss_list)) if val_loss_list else None
        val_mae_list = history.history.get('val_mae', [])
        best_mae = float(min(val_mae_list)) if val_mae_list else None

        compressed_weights = serialize_weights_compressed(model)

        training_time = time.time() - start_time
        mae_str = f"{best_mae:.4f}" if best_mae is not None else "N/A"
        print(f"{LOG} Foundation done {code}: val_loss={best_val_loss:.6f}, MAE={mae_str}, epochs={actual_epochs}/{epochs}, {training_time:.1f}s")

        tf.keras.backend.clear_session()

        result = {
            'success': True,
            'weightsB64': compressed_weights,
            'compressed': True,
            'epochs': actual_epochs,
            'trainingTime': round(training_time, 2)
        }
        if best_val_loss is not None:
            result['val_loss'] = round(best_val_loss, 6)
        if best_mae is not None:
            result['mae'] = round(best_mae, 6)

        return result

    except Exception as e:
        training_time = time.time() - start_time
        print(f"{LOG} Error in foundation training {code}: {str(e)}")
        tf.keras.backend.clear_session()
        return {
            'success': False,
            'error': str(e),
            'trainingTime': round(training_time, 2)
        }


def handler(event):
    start_time = time.time()
    input_data = event['input']

    code = input_data.get('code')
    commodity = input_data.get('commodity')
    config = input_data.get('config', {})

    if not code or not commodity:
        return {'success': False, 'error': 'Missing required fields: code, commodity'}

    if config.get('mode') == 'foundation':
        return handler_foundation(input_data, config, start_time)

    is_compressed = input_data.get('compressed', False)
    if is_compressed:
        print(f"{LOG} Payload is zlib-compressed")

    if input_data.get('featuresB64') and input_data.get('featuresShape'):
        feat_data = decode_base64_to_float32(input_data['featuresB64'], is_compressed)
        x_data = np.reshape(feat_data, input_data['featuresShape'])
    elif input_data.get('features'):
        x_data = np.array(input_data['features'], dtype=np.float32)
    else:
        return {'success': False, 'error': 'Missing features data'}

    if input_data.get('targetsB64') and input_data.get('targetsShape'):
        targ_data = decode_base64_to_float32(input_data['targetsB64'], is_compressed)
        y_data = np.reshape(targ_data, input_data['targetsShape'])
    elif input_data.get('targets'):
        y_data = np.array(input_data['targets'], dtype=np.float32)
    else:
        return {'success': False, 'error': 'Missing targets data'}

    window_size = x_data.shape[1]
    feature_count = x_data.shape[2]
    epochs = config.get('epochs', 10)
    learning_rate = config.get('learningRate', 0.0005)
    batch_size = min(config.get('batchSize', 32), max(16, len(x_data) // 8))

    print(f"{LOG} Training {code} ({commodity}), shape={list(x_data.shape)}, targets={list(y_data.shape)}, epochs={epochs}, batch={batch_size}")

    try:
        model = build_model(commodity, feature_count, window_size, config)

        f_weights = get_foundation_weights(input_data)
        if f_weights and len(f_weights) > 0:
            m_weights = model.get_weights()
            to_set = []
            min_len = min(len(f_weights), len(m_weights))
            for i in range(len(m_weights)):
                if i < min_len and list(f_weights[i].shape) == list(m_weights[i].shape):
                    to_set.append(f_weights[i])
                else:
                    to_set.append(m_weights[i])
            model.set_weights(to_set)

            for layer in model.layers:
                if isinstance(layer, LSTM):
                    layer.trainable = False
            print(f"{LOG} Foundation weights applied, LSTM layers frozen")

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss='mean_squared_error',
            metrics=['mae']
        )

        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor='val_mae',
            patience=3,
            restore_best_weights=True,
            verbose=0
        )

        history = model.fit(
            x_data, y_data,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            verbose=0,
            callbacks=[early_stop]
        )

        actual_epochs = len(history.history['loss'])
        val_mae = history.history.get('val_mae')
        mae_list = history.history.get('mae')
        best_mae = float(min(val_mae)) if val_mae else float(min(mae_list))
        last_mae = best_mae

        trained_weights = serialize_weights(model)

        training_time = time.time() - start_time
        print(f"{LOG} Done {code}: MAE={last_mae:.4f}, epochs={actual_epochs}/{epochs}, {training_time:.1f}s")

        tf.keras.backend.clear_session()

        return {
            'success': True,
            'weights': trained_weights,
            'mae': round(last_mae, 6),
            'epochs': actual_epochs,
            'trainingTime': round(training_time, 2)
        }

    except Exception as e:
        training_time = time.time() - start_time
        print(f"{LOG} Error training {code}: {str(e)}")
        tf.keras.backend.clear_session()
        return {
            'success': False,
            'error': str(e),
            'trainingTime': round(training_time, 2)
        }


if __name__ == '__main__':
    print(f"{LOG} Starting RunPod Serverless worker...")
    gpus = tf.config.list_physical_devices('GPU')
    print(f"{LOG} GPUs available: {len(gpus)}")
    for gpu in gpus:
        print(f"{LOG}   {gpu}")
        tf.config.experimental.set_memory_growth(gpu, True)
    runpod.serverless.start({'handler': handler})
