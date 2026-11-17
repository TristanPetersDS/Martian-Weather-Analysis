# Martian Weather Analysis: forecasting and anomaly detection with Curiosity REMS data

This project models daily ground temperature at Gale Crater from the Curiosity
rover's Rover Environmental Monitoring Station (REMS). It cleans and explores
nine Earth years of daily summaries, compares three forecasting approaches
(SARIMA, SARIMAX and an LSTM), and uses LSTM autoencoders on the forecast
residuals to flag unusual stretches of weather.

It was my capstone for Springboard's Data Science Career Track (2025). The
known weaknesses of the evaluation are listed under
[Known limitations](#known-limitations), with the fixes I am making.

## Data

- **Source:** REMS daily summaries, from the
  [Kaggle mirror](https://www.kaggle.com/datasets/deepcontractor/mars-rover-environmental-monitoring-station/data)
  of the [NASA REMS archive](https://atmos.nmsu.edu/data_and_services/atmospheres_data/MARS/curiosity/rems.html).
- **Coverage:** 3,197 daily records from sol 1 (7 Aug 2012) to sol 3368
  (26 Jan 2022). The other 171 sols have no record; the longest gap is 29
  sols. The gaps are listed in `data/cleaned/data_gaps.csv`.
- **Variables used:** solar longitude (Ls), daily maximum and minimum ground
  and air temperature (°C), mean pressure (Pa), sunrise and sunset (from
  which day length is derived) and the UV index (an ordinal category). Wind
  speed and humidity are empty in this release and are dropped.
- **Targets:** `avg_ground_temp` and `avg_air_temp` are the mid-range
  (max + min) / 2 of each sol, not a true daily mean.

## Method

1. **Cleaning** (`01_data_cleaning.ipynb`). "Value not available" becomes
   missing. Missing temperatures and pressures within recorded sols are
   linearly interpolated, and missing UV values take the most common value in
   a 30-record window. Martian month and season are derived from Ls, and day
   length from sunrise and sunset. The 171 missing sols are detected but not
   filled.
2. **Exploration** (`02_eda.ipynb`). Seasonal and inter-annual cycles,
   correlations, stationarity tests and rolling statistics.
3. **Preprocessing** (`03_preprocessing.ipynb`). Ls is encoded as sine and
   cosine, season and UV as dummy variables, and the numeric columns are
   min-max scaled. The data are split in time: sols 1–2694 for training
   (2,557 records) and sols 2695–3368 for testing (640 records).
4. **Modelling** (`04_modeling.ipynb`).
   - **SARIMA** on ground temperature alone, forecasting all 640 test
     records from the end of the training period.
   - **SARIMAX**, the same with exogenous regressors: same-day air
     temperature, pressure, day length, UV, season and Ls.
   - **LSTM** predicting each test sol from the 60 records before it, with a
     second output that classifies the change as up, flat or down.
   - **Anomaly detection:** an LSTM autoencoder on 30-record windows of the
     SARIMA and of the SARIMAX test residuals, flagging windows whose
     reconstruction error exceeds the 95th percentile.

## Results

Test-set errors in ground temperature, from
`notebooks/model_outputs/model_metrics_forecasters.csv`:

| Model | What it is asked to do | MAE (°C) | RMSE (°C) | Direction accuracy |
|---|---|---|---|---|
| SARIMA | Forecast 640 records ahead from ground temperature alone | 3.57 | 4.52 | 44.0% |
| SARIMAX | Same, but given each test sol's measured air temperature, pressure and other regressors | 2.10 | 2.91 | 50.2% |
| LSTM | Predict one sol ahead from the 60 measured records before it (scored on 580 records) | 1.77 | 2.94 | 49.7% |

The LSTM has the lowest MAE and SARIMAX the lowest RMSE, but the table does
not rank the methods, because the three rows answer different questions.
SARIMA forecasts nearly two Earth years ahead; SARIMAX is given same-day
measurements, including air temperature, that a real forecast would not have;
the LSTM only ever predicts one step ahead. Direction accuracy is defined
differently for the LSTM (its three-class output) than for the ARIMA models
(the sign of each predicted change), and all three values are close to 50%,
so none of the models shows skill at the day-to-day direction.

The two autoencoders each flag 31 of 610 windows, which is the 5% that a
95th-percentile threshold flags by construction. The flagged windows have not
been compared with known atmospheric events, so they are candidates for a
closer look rather than detections.

## Known limitations

I found these while reviewing the project after the course. They are being
fixed in the order listed; this README will be updated as each lands.

1. **Hyperparameters were chosen on the test set.** SARIMA and SARIMAX were
   each picked from 476 configurations by test-set RMSE, and the LSTM was
   picked from 288 configurations by test-set error, with early stopping also
   watching the test set. The errors above are therefore optimistic. Fix: select on a validation period carved from the end of the
   training data, then score the test set once.
2. **No naive baseline.** Without persistence, seasonal-naive (one Martian
   year, 668 sols, earlier) and Ls-climatology forecasts on the same test
   points, the errors cannot be judged.
3. **The scaler saw the whole series.** Min-max scaling was fitted before the
   train/test split, a mild leak. (The project report in `reports/` says the
   opposite; it is wrong on this point.)
4. **Models are not compared on a common task.** See Results. Fix: evaluate
   every model at the same horizons, and give SARIMAX only regressors that
   are known in advance, such as Ls.
5. **The annual cycle is missing from SARIMA.** Its seasonal periods were
   searched over 5–35 sols, while the dominant cycle is the 668-sol Martian
   year. Much of its larger error is that unmodelled cycle.
6. **Rows are treated as evenly spaced.** With 171 sols missing, a 60-record
   window can span more than 60 sols, and the sol numbers attached to flagged
   anomaly windows drift by the number of gaps before them.
7. **The anomaly threshold is circular.** Each autoencoder is trained,
   scored and thresholded on the same test residuals. Fix: train on
   training-period residuals, set the threshold on a separate period, then
   score the test period.
8. **Two EDA conclusions do not follow from the tests shown.** The Granger
   test in `02_eda.ipynb` tests whether pressure helps predict ground
   temperature, but the text reads it the other way round; and the
   correlation p-values assume independent samples, which daily weather is
   not.
9. **No random seeds are set**, so the neural-network results cannot be
   reproduced exactly.

The report and slides in `reports/` predate these corrections and will be
regenerated.

## Project structure

```plaintext
├── data/
│   ├── raw/                     # REMS dataset as downloaded from Kaggle
│   ├── cleaned/                 # Cleaned data and the list of missing sols
│   └── processed/               # Encoded, scaled and split datasets, and the fitted scaler
├── notebooks/
│   ├── 01_data_cleaning.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_preprocessing.ipynb
│   ├── 04_modeling.ipynb
│   └── model_outputs/           # Saved predictions, residuals and metrics
├── outputs/plots/               # Figures exported by the notebooks
├── reports/
│   ├── model_metrics.md         # Model configurations and metrics
│   ├── project_report.pdf       # Capstone report
│   └── Capstone 2 - Presentation.pdf
├── utils/
│   ├── hybrid_anomaly_util.py   # Residual sequences, autoencoder training and anomaly flags
│   ├── time_handler.py          # Martian time conversion
│   └── utilities.py             # Loading, cleaning and plotting helpers
└── LICENSE
```

The fitted models are not in the repository; notebook 04 refits them into a
`models/` folder that is not committed. It also loads a target scaler
(`models/y_target_scaler.pkl`) that no notebook creates, so its LSTM section
does not yet run from a fresh clone.

## Tools

Python, pandas, NumPy, Matplotlib, seaborn, statsmodels (SARIMA, SARIMAX,
stationarity and Granger tests), TensorFlow/Keras (LSTM forecaster and
autoencoders) and scikit-learn (scaling and metrics).

## Author

**Tristan Peters** — M.S. Physics ·
[LinkedIn](https://www.linkedin.com/in/tristan-peters-ds/) ·
[GitHub](https://github.com/TristanPetersDS)

## License

The code is released under the MIT License. The REMS data are NASA's; see
the source links above.
