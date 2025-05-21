
# 🌌 Martian Weather Analysis: Forecasting & Anomaly Detection Using Curiosity REMS Data

## 📖 Overview

This project explores Martian atmospheric behavior using data from the **REMS (Rover Environmental Monitoring Station)** onboard NASA's **Curiosity rover**, stationed at Gale Crater. The analysis focuses on forecasting **average ground temperature** using time series modeling, and identifying **atypical atmospheric events** through residual-based anomaly detection.

The full workflow includes:

- Structured **data cleaning and temporal engineering**
- Insight-driven **exploratory data analysis**
- Modular **data preprocessing** for multiple model pipelines
- Statistical and deep learning-based **forecasting and anomaly detection**

This work demonstrates how data collected on another planet can be processed and modeled to support scientific understanding and future mission planning.

## 🚀 Objectives

- 🧼 Clean and prepare REMS atmospheric data for analysis.
- 🔍 Uncover temporal and seasonal trends in Martian weather through EDA.
- 🧪 Build robust time series models to forecast ground temperature.
- ⚠️ Detect anomalies using hybrid residual analysis and deep learning.
- 📊 Evaluate and compare model performance across strategies.

## 📁 Project Structure

```plaintext
├── data/                         # Raw and processed REMS data files
│   ├── raw/                      # Original REMS dataset from Kaggle
│   ├── cleaned/                  # Cleaned and gap-imputed data
│   └── processed/                # Scaled, encoded, and feature-rich datasets
│
├── notebooks/                    # Jupyter notebooks for each analysis stage
│   ├── model_outputs/           # Saved predictions, residuals, metrics, and models
│   ├── 01_data_cleaning.ipynb   # Handle missing values and temporal structuring
│   ├── 02_eda.ipynb             # Explore trends and seasonal patterns
│   ├── 03_preprocessing.ipynb   # Feature engineering and ML-ready preparation
│   └── 04_modeling.ipynb        # Forecasting and anomaly detection workflow
│
├── outputs/
│   └── plots/                   # Exported visualizations and figures
│
├── utils/                       # Python utility scripts
│   ├── hybrid_anomaly_util.py   # Residual scoring + LSTM anomaly detection
│   ├── time_handler.py          # Martian time conversion and handling
│   ├── time_utils.py            # Time encoding and utility helpers
│   └── utilities.py             # General-purpose functions (loading, cleaning, I/O)
│
└── README.md                    # Project overview (this file)

```

## 🛰️ Dataset

This project uses environmental sensor data collected by NASA’s **Curiosity rover** via the **Rover Environmental Monitoring Station (REMS)** at Gale Crater on Mars.

- **Primary Source**: [NASA REMS archive](https://atmos.nmsu.edu/data_and_services/atmospheres_data/MARS/curiosity/rems.html)  
- **Working Dataset**: [Kaggle mirror by DEEP CONTRACTOR](https://www.kaggle.com/datasets/deepcontractor/mars-rover-environmental-monitoring-station/data)  
- **Total Records**: 3,197 sols (Martian days)

### 🔑 Key Variables
- `Ls` – Solar longitude (0°–360°): tracks Mars’ position in orbit
- `max_ground_temp`, `min_ground_temp` – Ground temperature extremes (°C)
- `max_air_temp`, `min_air_temp` – Atmospheric temperature extremes (°C)
- `mean_pressure` – Daily average atmospheric pressure (Pa)
- `UV_Radiation` – Discrete UV index (e.g., low, moderate, high)
- `sunrise`, `sunset` – Time of solar events (Mars local time)
- `weather` – Qualitative description (e.g., Sunny, Cloudy)
- `earth_date_time` – Earth timestamp of the sol
- `sol_number` – Martian sol index

### 🧪 Derived Features
- `mars_year`, `mars_month`, `mars_season` – Computed from Ls to align with the Martian calendar
- `avg_ground_temp`, `avg_air_temp` – Added as composite targets for forecasting

### 🧼 Missing Data Strategy
- `"Value not available"` strings replaced with `NaN`
- Feature and index-level gaps detected via:
  - **Rolling mode imputation** (categorical)
  - **Linear interpolation** (continuous)
- Time index continuity validated to support forecasting and anomaly detection

## 📊 Methodology

### 1. Data Cleaning
- Standardized all numeric columns
- Derived Martian calendar features
- Identified and imputed index and feature-level gaps

### 2. Exploratory Analysis
- Investigated seasonal and yearly cycles in weather data
- Assessed feature correlations and temporal dependencies
- Highlighted outliers and environmental irregularities

### 3. Preprocessing
- Encoded cyclical features (e.g., `Ls`) with sine/cosine
- Created dummy variables for season and UV categories
- Scaled datasets for compatibility with ML and DL models

### 4. Modeling & Anomaly Detection
- **SARIMA/SARIMAX** models for univariate and multivariate forecasting
- **LSTM Autoencoder** for residual-based anomaly detection
- Combined statistical and deep learning models into a hybrid anomaly pipeline

## 📈 Results Summary

- SARIMAX performed best in terms of temperature forecasting MAE.
- Anomaly detection pipeline captured sharp deviations in residuals across solar longitude and pressure conditions.
- Detected events aligned with seasonal transitions and UV radiation shifts, suggesting scientific relevance.

## 🧰 Tools & Technologies

- Python (Pandas, NumPy, Matplotlib, Seaborn)
- Statsmodels (SARIMA/SARIMAX)
- TensorFlow / Keras (LSTM Autoencoder)
- Scikit-learn (Preprocessing & Metrics)
- Jupyter Notebooks

## 📚 Future Work

- Create pipeline to convert up-to-date archived REMS data directly into format suitable for this analysis
- Recreate analysis with archive data directly

## 🧑‍🚀 Author

**Tristan Peters**

M.S. Physics | Data Scientist | [LinkedIn](https://www.linkedin.com/in/tristan-peters-ds/) | [GitHub](https://github.com/TristanPetersDS)

## 📄 License

This project is open source under the MIT License.
