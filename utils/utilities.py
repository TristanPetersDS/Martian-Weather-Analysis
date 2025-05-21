import os
import json
import joblib

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.api import qqplot
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from typing import Optional, Tuple, Dict, Union

from tensorflow.keras.utils import Sequence

# ─────────────────────────────────────────────────────────────────────────────
# Data Generators
# ─────────────────────────────────────────────────────────────────────────────

class CombinedSequence(Sequence):
    """
    Keras sequence to merge regression and classification label generators.

    Parameters:
        reg_gen: Generator yielding regression targets.
        clf_gen: Generator yielding classification targets.
    """
    def __init__(self, reg_gen, clf_gen):
        self.reg_gen = reg_gen
        self.clf_gen = clf_gen
        assert len(self.reg_gen) == len(self.clf_gen)

    def __len__(self):
        return len(self.reg_gen)

    def __getitem__(self, idx):
        x, y_reg = self.reg_gen[idx]
        _, y_clf = self.clf_gen[idx]
        return x, {"forecast": y_reg, "direction": y_clf}


# ─────────────────────────────────────────────────────────────────────────────
# Model Output I/O
# ─────────────────────────────────────────────────────────────────────────────

def load_model_outputs(
    model_name: str,
    load_dir: str = "model_outputs/",
    suffix: Optional[str] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[Dict[str, float]]]:
    """
    Load saved predictions, residuals, and optional metrics for a model.

    Parameters:
        model_name (str): Model identifier used in filenames.
        load_dir (str): Directory to load model outputs from.
        suffix (str): Optional version suffix for distinguishing saved files.

    Returns:
        Tuple of (y_true, y_pred, residuals, metrics)
    """
    tag = model_name.lower()
    if suffix:
        tag += f"_{suffix}"

    def _load_npy(filename):
        path = os.path.join(load_dir, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")
        return np.load(path)

    y_true = _load_npy(f"y_true_{tag}.npy")
    y_pred = _load_npy(f"y_pred_{tag}.npy")
    residuals = _load_npy(f"residuals_{tag}.npy")

    metrics = None
    metrics_path = os.path.join(load_dir, f"metrics_{tag}.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            metrics = json.load(f)

    print(f"Loaded outputs for {model_name} from {load_dir}")
    return y_true, y_pred, residuals, metrics


def save_model_outputs(
    model_name: str,
    y_true: Union[np.ndarray, pd.Series],
    y_pred: Union[np.ndarray, pd.Series],
    save_dir: str = "model_outputs/",
    metrics: Optional[Dict[str, float]] = None,
    suffix: Optional[str] = None
):
    """
    Save predictions, residuals, and evaluation metrics for a model.

    Parameters:
        model_name (str): Model identifier used in filenames.
        y_true (array): Ground truth values.
        y_pred (array): Model predictions.
        save_dir (str): Directory to save outputs to.
        metrics (dict): Optional evaluation results to save.
        suffix (str): Optional version suffix for distinguishing saved files.
    """
    os.makedirs(save_dir, exist_ok=True)

    tag = model_name.lower()
    if suffix:
        tag += f"_{suffix}"

    np.save(os.path.join(save_dir, f"y_pred_{tag}.npy"), y_pred)
    np.save(os.path.join(save_dir, f"y_true_{tag}.npy"), y_true)
    np.save(os.path.join(save_dir, f"residuals_{tag}.npy"), y_true - y_pred)

    if metrics:
        with open(os.path.join(save_dir, f"metrics_{tag}.json"), "w") as f:
            json.dump(metrics, f, indent=4)

    print(f"Saved outputs for {model_name} to {save_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# Data Quality + Imputation
# ─────────────────────────────────────────────────────────────────────────────

def check_missing(df, unavailable_value="Value not available"):
    """
    Summarize missing and unavailable values in a DataFrame.

    Parameters:
        df (DataFrame): Input dataset.
        unavailable_value (str): Placeholder string indicating unavailable data.

    Returns:
        DataFrame with missing and unavailable value counts and percentages.
    """
    result = pd.concat([
        df.isnull().sum(),
        df.eq(unavailable_value).sum(),
        100 * (df.isnull().mean() + df.eq(unavailable_value).mean())
    ], axis=1)

    result.columns = ['NaN Count', f'"{unavailable_value}" Count', 'Total % Missing or Unavailable']
    return result.sort_values(by='Total % Missing or Unavailable', ascending=False)


def gap_size(df):
    """
    Identify and summarize NaN gaps in columns and missing index ranges.

    Parameters:
        df (DataFrame): Input dataset.

    Returns:
        DataFrame summarizing gap size and positions per column or index.
    """
    gap_data = []

    # Index gap detection
    if df.index.is_monotonic_increasing and pd.api.types.is_integer_dtype(df.index):
        full_index = pd.RangeIndex(start=df.index.min(), stop=df.index.max() + 1)
        missing_index = full_index.difference(df.index)
        if len(missing_index) > 0:
            is_gap = (missing_index.to_series().diff() != 1).cumsum()
            for _, group in missing_index.to_series().groupby(is_gap):
                gap_data.append({
                    'Column': 'Index',
                    'Gap Size': len(group),
                    'Start Index': group.iloc[0],
                    'End Index': group.iloc[-1]
                })

    # Column gap detection
    for col in df.columns:
        is_null = df[col].isnull()
        gap_groups = (is_null != is_null.shift()).cumsum()
        for _, group_df in df.loc[is_null].groupby(gap_groups):
            gap_data.append({
                'Column': col,
                'Gap Size': len(group_df),
                'Start Index': group_df.index[0],
                'End Index': group_df.index[-1]
            })

    gap_df = pd.DataFrame(gap_data)
    if gap_df.empty:
        print("No gaps found.")
        return gap_df

    return gap_df.sort_values(by=['Column', 'Start Index']).reset_index(drop=True)


def rolling_mode(series, window=30):
    """
    Fill missing values using the mode of a rolling window.

    Parameters:
        series (Series): Input time series.
        window (int): Rolling window size for mode calculation.

    Returns:
        Series with missing values filled by rolling mode.
    """
    def mode_func(x):
        modes = x.mode()
        return modes[0] if not modes.empty else pd.NA

    return series.fillna(series.rolling(window, center=True, min_periods=1).apply(mode_func, raw=False))


# ─────────────────────────────────────────────────────────────────────────────
# Feature Engineering
# ─────────────────────────────────────────────────────────────────────────────

def map_season(degree):
    """
    Map solar longitude (Ls) to Martian season.

    Parameters:
        degree (float): Solar longitude in degrees [0–360).

    Returns:
        String representing Martian season.
    """
    if pd.isnull(degree):
        return np.nan
    elif 0 <= degree < 90:
        return 'autumn'
    elif 90 <= degree < 180:
        return 'winter'
    elif 180 <= degree < 270:
        return 'spring'
    elif 270 <= degree < 360:
        return 'summer'


def map_months(degree):
    """
    Map solar longitude (Ls) to a 12-month Martian calendar.

    Parameters:
        degree (float): Solar longitude in degrees [0–360).

    Returns:
        Integer from 1 to 12 representing Martian month.
    """
    if pd.isnull(degree):
        return np.nan
    return (int(degree // 30) + 1) if 0 <= degree < 360 else np.nan


def calculate_year_column(df, ls_column):
    """
    Assign Martian year numbers based on Ls reset points.

    Parameters:
        df (DataFrame): Mars dataset.
        ls_column (str): Column containing solar longitude (Ls).

    Returns:
        DataFrame with added 'year' column.
    """
    year = 1
    years = []
    prev_ls = 0

    for current_ls in df[ls_column]:
        if pd.notnull(current_ls):
            if prev_ls > current_ls:
                year += 1
            prev_ls = current_ls
        years.append(year)

    df['year'] = years
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Time Series Analysis
# ─────────────────────────────────────────────────────────────────────────────

def decompose_adf(df, feature, lag=200, period=668, model='additive', store=False):
    """
    Decompose a time series and run an ADF stationarity test.

    Parameters:
        df (DataFrame): Input time series data.
        feature (str): Target column for analysis.
        lag (int): Lags for ACF plot.
        period (int): Seasonal period (e.g., 668 sols for Mars).
        model (str): Decomposition type ('additive' or 'multiplicative').
        store (bool): If True, return results dictionary; otherwise, print summary.

    Returns:
        Optional dictionary with trend, seasonal, residual, and ADF results.
    """
    try:
        result = seasonal_decompose(df[feature], model=model, period=period, extrapolate_trend='freq')
        trend = result.trend
        seasonal = result.seasonal
        residual = result.resid.dropna()

        result.plot(); plt.show()
        residual.hist(bins=min(len(residual) // 10, 50), figsize=(8, 6), alpha=0.7)
        plt.title(f"Histogram of Residuals for {feature}"); plt.show()
        plot_acf(residual, lags=lag); plt.title(f"Autocorrelation for {feature}"); plt.show()

        dftest = adfuller(residual, autolag='AIC')
        if store:
            return {
                "trend": trend,
                "seasonal": seasonal,
                "residual": residual,
                "ADF Test": {
                    "ADF Statistic": dftest[0],
                    "P-Value": dftest[1],
                    "Num of Lags": dftest[2],
                    "Num of Observations": dftest[3],
                    "Critical Values": dftest[4]
                }
            }
        else:
            print(f"\nADF Test Results for {feature}:")
            print(f"ADF Statistic: {dftest[0]}\nP-Value: {dftest[1]}")
            print(f"Lags Used: {dftest[2]}, Observations: {dftest[3]}")
            print("Critical Values:")
            for key, val in dftest[4].items():
                print(f"\t{key}: {val}")
    except Exception as e:
        print(f"Error in decompose_adf: {e}")
        return None
