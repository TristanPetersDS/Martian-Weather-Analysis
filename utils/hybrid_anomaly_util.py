import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, RepeatVector, TimeDistributed, Dense
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

def create_lstm_autoencoder(seq_len, n_features):
    inp = Input(shape=(seq_len, n_features))
    x   = LSTM(32, return_sequences=False)(inp)
    x   = RepeatVector(seq_len)(x)
    x   = LSTM(32, return_sequences=True)(x)
    out = TimeDistributed(Dense(n_features))(x)
    
    model = Model(inputs=inp, outputs=out)
    model.compile(optimizer='adam', loss='mae')
    return model

def prepare_residual_sequences(residuals, seq_len=30):
    residuals = residuals.reshape(-1, 1)
    scaler = StandardScaler()
    res_scaled = scaler.fit_transform(residuals)
    
    gen = TimeseriesGenerator(res_scaled, res_scaled, length=seq_len, batch_size=32)
    return gen, scaler

def compute_reconstruction_errors(model, gen):
    # Predict using the generator
    preds = model.predict(gen, verbose=0)

    # Extract the true values in the correct shape
    true_values = []
    for _, y in gen:
        true_values.append(y)
    true = np.concatenate(true_values, axis=0)

    # Compute per-sequence MAE
    errors = np.mean(np.abs(preds - true), axis=(1, 2))
    return errors


def detect_anomalies(errors, threshold=None, percentile=95):
    if threshold is None:
        threshold = np.percentile(errors, percentile)
    return (errors > threshold).astype(int), threshold

from tensorflow.keras.callbacks import Callback
from tqdm.notebook import tqdm

class TQDMProgressBar(Callback):
    def on_train_begin(self, logs=None):
        self.epochs = self.params['epochs']
        self.progress_bar = tqdm(total=self.epochs, desc='Training Progress', unit='epoch')

    def on_epoch_end(self, epoch, logs=None):
        self.progress_bar.update(1)
        loss = logs.get('loss')
        self.progress_bar.set_postfix({'loss': f'{loss:.4f}'})

    def on_train_end(self, logs=None):
        self.progress_bar.close()