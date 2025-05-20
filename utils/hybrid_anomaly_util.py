import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, RepeatVector, TimeDistributed, Dense
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.callbacks import Callback
from tqdm.notebook import tqdm


# ─────────────────────────────────────────────────────────────────────────────
# LSTM Autoencoder Model
# ─────────────────────────────────────────────────────────────────────────────

def create_lstm_autoencoder(seq_len, n_features):
    """
    Build and compile an LSTM-based autoencoder for sequence reconstruction.
    """
    inp = Input(shape=(seq_len, n_features))
    x   = LSTM(32, return_sequences=False)(inp)
    x   = RepeatVector(seq_len)(x)
    x   = LSTM(32, return_sequences=True)(x)
    out = TimeDistributed(Dense(n_features))(x)

    model = Model(inputs=inp, outputs=out)
    model.compile(optimizer='adam', loss='mae')
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Data Preparation
# ─────────────────────────────────────────────────────────────────────────────

def prepare_residual_sequences(residuals, seq_len=30):
    """
    Scale residuals and wrap them in a Keras TimeseriesGenerator.
    """
    residuals = residuals.reshape(-1, 1)
    scaler = StandardScaler()
    res_scaled = scaler.fit_transform(residuals)
    gen = TimeseriesGenerator(res_scaled, res_scaled, length=seq_len, batch_size=32)
    return gen, scaler


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation and Anomaly Detection
# ─────────────────────────────────────────────────────────────────────────────

def compute_reconstruction_errors(model, gen, batch_size=32):
    """
    Compute mean absolute reconstruction error for each input sequence.
    """
    errors = []

    for X_batch, y_batch in tqdm(gen, desc="Computing reconstruction errors"):
        pred_batch = model.predict(X_batch, verbose=0)
        batch_error = np.mean(np.abs(pred_batch - y_batch), axis=(1, 2))
        errors.extend(batch_error)

    return np.array(errors)


def detect_anomalies(errors, threshold=None, percentile=95):
    """
    Detect anomalies based on a reconstruction error threshold.
    """
    if threshold is None:
        threshold = np.percentile(errors, percentile)
    return (errors > threshold).astype(int), threshold


# ─────────────────────────────────────────────────────────────────────────────
# Keras Utilities
# ─────────────────────────────────────────────────────────────────────────────

class TQDMProgressBar(Callback):
    """
    Keras callback for tqdm-based training progress display.
    """
    def on_train_begin(self, logs=None):
        self.epochs = self.params['epochs']
        self.progress_bar = tqdm(total=self.epochs, desc='Training Progress', unit='epoch')

    def on_epoch_end(self, epoch, logs=None):
        self.progress_bar.update(1)
        loss = logs.get('loss')
        self.progress_bar.set_postfix({'loss': f'{loss:.4f}'})

    def on_train_end(self, logs=None):
        self.progress_bar.close()


# ─────────────────────────────────────────────────────────────────────────────
# Custom Sequence Generator
# ─────────────────────────────────────────────────────────────────────────────

class AutoencoderSequenceGenerator(tf.keras.utils.Sequence):
    """
    Custom sequence generator for training an autoencoder on time series data.
    """
    def __init__(self, data, seq_len=30, batch_size=32):
        self.data = data
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.indices = np.arange(len(data) - seq_len)
        if len(self.indices) == 0:
            raise ValueError(f"Insufficient data to form a single sequence of length {seq_len}")

    def __len__(self):
        return int(np.ceil(len(self.indices) / self.batch_size))

    def __getitem__(self, idx):
        start = idx * self.batch_size
        end = min((idx + 1) * self.batch_size, len(self.indices))
        if start >= end:
            raise IndexError(f"Batch {idx} out of bounds (start={start}, end={end})")

        batch_idx = self.indices[start:end]
        X = np.array([self.data[i : i + self.seq_len] for i in batch_idx])
        return X, X  # Autoencoder: input == target
