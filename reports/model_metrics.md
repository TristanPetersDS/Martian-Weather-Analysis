# 📈 Forecasting Models

## ✅ Best SARIMA Configuration

| Parameter        | Value               |
|------------------|---------------------|
| Seasonal Period  | 25                  |
| Order            | (2, 0, 2)           |
| Seasonal Order   | (0, 1, 1) + (25)    |
| AIC              | 10317.75            |
| MAE              | 3.57                |
| RMSE             | 4.52                |

**Notes:**
- Captures general trend well.
- Struggles with sharp transitions.

---

## ✅ Best SARIMAX Configuration

| Parameter        | Value               |
|------------------|---------------------|
| Seasonal Period  | 5                   |
| Order            | (0, 0, 1)           |
| Seasonal Order   | (1, 1, 1) + (5)     |
| AIC              | 10526.86            |
| MAE              | 2.10                |
| RMSE             | 2.91                |

**Notes:**
- Improved short-term accuracy using exogenous features.

---

## ✅ Best LSTM Configuration

| Parameter            | Value                        |
|----------------------|------------------------------|
| Feature Set          | Univariate (`avg_ground_temp`) |
| Sequence Length      | 60                           |
| Batch Size           | 32                           |
| LSTM Units           | 20                           |
| Dropout Rate         | 0.10                         |
| Epochs               | 30                           |
| Early Stopping       | Yes (patience = 3)           |
| Validation Split     | 0.2                          |
| Optimizer            | Adam                         |
| Loss Functions       | MSE (forecast), CCE (direction) |
| Metrics              | MAE (forecast), Accuracy (direction) |

| Metric              | Value         |
|---------------------|---------------|
| MAE                 | 1.768 °C      |
| RMSE                | 2.939 °C      |
| sMAPE               | 4.16 %        |
| Direction Accuracy  | 49.66 %       |
| Selection Score     | 2.775         |

**Architecture Summary:**
- Multitask LSTM model with shared backbone.
- Input: (30, 1) sequence of `avg_ground_temp`
- LSTM (units=20, tanh, return_sequences=False)
- Dropout (0.3)
- Dual Heads:
  - Forecast → Dense(1), linear
  - Direction → Dense(3), softmax

**Notes:**
- Selection Score = MAE + 0.02 × (100 - Direction Accuracy)
- Prioritizes local pattern responsiveness; may risk overfitting.
- Direction classification performance was weaker than expected.

---

# ⚠️ Anomaly Detection (Hybrid LSTM Autoencoders)

| Attribute         | Value                                                        |
|-------------------|--------------------------------------------------------------|
| Model Type        | LSTM Autoencoder                                             |
| Input             | Residuals from SARIMA and SARIMAX forecasts                  |
| Training Data     | Sequences of forecast residuals                              |
| Purpose           | Learn baseline error patterns; flag deviations               |
| Anomaly Threshold | 95th percentile of reconstruction error                      |

### 🔍 Reconstruction Error Metrics

| Metric              | SARIMA AE | SARIMAX AE |
|---------------------|-----------|------------|
| Mean Error          | 0.4039    | 0.5262     |
| Median Error        | 0.3620    | 0.4664     |
| 95th Percentile     | 0.8051    | 0.9044     |
| Max Error           | 0.9418    | 1.1289     |
| Std Deviation       | 0.1788    | 0.2052     |

**Notes:**
- Reconstruction error is calculated as MAE between true and reconstructed residual sequences.
- Anomalies are sols with reconstruction error > 95th percentile.
- SARIMA-based autoencoder showed lower variance; SARIMAX AE detected more high-error spikes.

---

