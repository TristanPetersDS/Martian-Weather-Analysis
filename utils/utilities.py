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


def load_model_outputs(
    model_name: str,
    load_dir: str = "model_outputs/",
    suffix: Optional[str] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[Dict[str, float]]]:
    """
    Load saved prediction outputs and residuals for a model.
    
    Parameters:
        model_name (str): Name of the model (e.g. "LSTM", "SARIMAX")
        load_dir (str): Directory to load from (default: "model_outputs/")
        suffix (str): Optional suffix used in saved filenames (e.g. "v2")

    Returns:
        Tuple containing:
            - y_true (np.ndarray)
            - y_pred (np.ndarray)
            - residuals (np.ndarray)
            - metrics (dict or None)
    """
    tag = model_name.lower()
    if suffix:
        tag += f"_{suffix}"

    def _load_npy(filename):
        path = os.path.join(load_dir, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")
        return np.load(path)

    # Load arrays
    y_true = _load_npy(f"y_true_{tag}.npy")
    y_pred = _load_npy(f"y_pred_{tag}.npy")
    residuals = _load_npy(f"residuals_{tag}.npy")

    # Load metrics if present
    metrics_path = os.path.join(load_dir, f"metrics_{tag}.json")
    metrics = None
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
    Save prediction outputs and evaluation metadata for a forecasting model.
    
    Parameters:
        model_name (str): Name of the model (e.g. "LSTM", "SARIMAX")
        y_true (array): Ground truth test set values
        y_pred (array): Model predictions
        save_dir (str): Base directory to save results into
        metrics (dict): Optional dict of evaluation metrics (e.g. {"mae": 1.2})
        suffix (str): Optional suffix to distinguish model versions (e.g. "v2")
    """
    os.makedirs(save_dir, exist_ok=True)
    
    tag = model_name.lower()
    if suffix:
        tag += f"_{suffix}"
    
    # Save predictions and residuals
    np.save(os.path.join(save_dir, f"y_pred_{tag}.npy"), y_pred)
    np.save(os.path.join(save_dir, f"y_true_{tag}.npy"), y_true)
    residuals = y_true - y_pred
    np.save(os.path.join(save_dir, f"residuals_{tag}.npy"), residuals)
    
    # Save metrics if provided
    if metrics:
        meta_path = os.path.join(save_dir, f"metrics_{tag}.json")
        with open(meta_path, "w") as f:
            json.dump(metrics, f, indent=4)

    print(f"Saved outputs for {model_name} to {save_dir}")


def check_missing(df, unavailable_value="Value not available"):
    """
    Check for missing values and a specified "unavailable" value in a DataFrame.
    
    Parameters:
        df (pd.DataFrame): The DataFrame to analyze.
        unavailable_value (str): The value representing unavailable data (default: "Value not available").
    
    Returns:
        pd.DataFrame: A DataFrame showing the counts and percentages of missing or unavailable values for each column.
    """
    result = pd.concat([
        df.isnull().sum(),  # Count of NaN values
        df.eq(unavailable_value).sum(),  # Count of "unavailable_value"
        100 * (df.isnull().mean() + df.eq(unavailable_value).mean())  # Percentage of missing or unavailable
    ], axis=1)
    
    result.columns = ['NaN Count', f'"{unavailable_value}" Count', 'Total % Missing or Unavailable']
    result = result.sort_values(by='Total % Missing or Unavailable', ascending=False)
    return result

def gap_size(df):
    """
    Calculate the sizes of gaps (sequential NaN entries) for all columns in a DataFrame,
    and gaps in the index if it is numerical, returning the information as a well-organized DataFrame.

    Parameters:
        df (pd.DataFrame): The DataFrame to analyze.

    Returns:
        pd.DataFrame: A DataFrame with columns ['Column', 'Gap Size', 'Start Index', 'End Index'],
                      sorted by 'Column' and 'Start Index'.
    """
    gap_data = []

    # Check for gaps in the index (if numerical)
    if df.index.is_monotonic_increasing and pd.api.types.is_integer_dtype(df.index):
        full_index = pd.RangeIndex(start=df.index.min(), stop=df.index.max() + 1)
        missing_index = full_index.difference(df.index)

        # Identify consecutive gaps in the missing index
        if len(missing_index) > 0:
            is_gap = (missing_index.to_series().diff() != 1).cumsum()
            for group_id, group in missing_index.to_series().groupby(is_gap):
                gap_size = len(group)
                start_index = group.iloc[0]
                end_index = group.iloc[-1]

                # Append index gap information
                gap_data.append({
                    'Column': 'Index',
                    'Gap Size': gap_size,
                    'Start Index': start_index,
                    'End Index': end_index
                })

    # Check for gaps in each column
    for column_name in df.columns:
        is_null = df[column_name].isnull()  # Boolean mask for NaN values
        gap_groups = (is_null != is_null.shift()).cumsum()  # Identify groups of consecutive NaNs

        # Iterate through each group to find NaN gaps
        for group_id, group_df in df.loc[is_null].groupby(gap_groups):
            gap_size = len(group_df)  # Number of NaN entries in this gap
            start_index = group_df.index[0]  # Starting index of the gap
            end_index = group_df.index[-1]  # Ending index of the gap

            # Append column gap information
            gap_data.append({
                'Column': column_name,
                'Gap Size': gap_size,
                'Start Index': start_index,
                'End Index': end_index
            })

    # Convert gap information to a DataFrame
    gap_df = pd.DataFrame(gap_data)

    # Handle case with no gaps
    if gap_df.empty:
        print("No gaps found in the DataFrame or its index.")
        return gap_df

    # Sort by column name and start index for better organization
    gap_df.sort_values(by=['Column', 'Start Index'], ascending=[True, True], inplace=True)
    gap_df.reset_index(drop=True, inplace=True)

    return gap_df




# Define a function to calculate the rolling mode
def rolling_mode(series, window=30):
    """
    Fill missing values in a series with the mode of a rolling window.
    
    Parameters:
        series (pd.Series): The input series with missing values (<NA>).
        window (int): The size of the rolling window.
        
    Returns:
        pd.Series: Series with missing values filled by the rolling mode.
    """
    # Define a helper function to compute mode
    def mode_func(x):
        modes = x.mode()
        return modes[0] if not modes.empty else pd.NA

    # Replace <NA> with the rolling mode
    filled = series.copy()
    filled = filled.fillna(series.rolling(window=window, center=True, min_periods=1).apply(mode_func, raw=False))
    return filled
    
# Define a function to map solar longitude to season
def map_season(degree):
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
        
# Define a function to map solar longitude to 12-month calendar
def map_months(degree):
    if pd.isnull(degree):
        return np.nan
    elif 0 <= degree < 30:
        return 1
    elif 30 <= degree < 60:
        return 2
    elif 60 <= degree < 90:
        return 3
    elif 90 <= degree < 120:
        return 4
    elif 120 <= degree < 150:
        return 5
    elif 150 <= degree < 180:
        return 6
    elif 180 <= degree < 210:
        return 7
    elif 210 <= degree < 240:
        return 8
    elif 240 <= degree < 270:
        return 9
    elif 270 <= degree < 300:
        return 10
    elif 300 <= degree < 330:
        return 11
    elif 330 <= degree < 360:
        return 12
    else:
        return np.nan  # Handle unexpected values outside 0-360 range

# Define a functuion to map solar longitude to year system beginning at the start of the mission with year 1
def calculate_year_column(df, ls_column):
    """
    Add a 'year' column based on the solar longitude (Ls) values.

    Parameters:
        df (pd.DataFrame): The Mars dataset.
        ls_column (str): The name of the column containing solar longitude (Ls).

    Returns:
        pd.DataFrame: The dataset with an added 'year' column.
    """
    # Initialize year with 1 for the first row
    year = 1
    years = []
    prev_ls = 0

    for current_ls in df[ls_column]:
        if pd.notnull(current_ls):  # Ensure the value is not NaN
            # Check if Ls has reset (crossing from ~360 back to 0)
            if prev_ls > current_ls:
                year += 1  # Increment year
            prev_ls = current_ls
        years.append(year)

    df['year'] 


def decompose_adf(df, feature, lag=200, period=668, model='additive', store=False):
    """
    Performs time series decomposition and ADF stationarity test on a given feature.

    Parameters:
    df (DataFrame): The input DataFrame containing the time series data.
    feature (str): The column name of the time series feature to analyze.
    period (int): The seasonal period for decomposition (default is 668 for Mars year).
    model (str): The type of decomposition model ('additive' or 'multiplicative').
    store (bool): If True, returns results as a dictionary; otherwise, only prints results.

    Returns:
    dict (optional): A dictionary containing trend, seasonal, residuals, and ADF test results.
    """

    try:
        # Perform seasonal decomposition
        decompose_result = seasonal_decompose(df[feature], model=model, period=period, extrapolate_trend='freq')

        # Extract components
        trend = decompose_result.trend
        seasonal = decompose_result.seasonal
        residual = decompose_result.resid.dropna()  # Drop NaN values for ADF test

        # Plot decomposition
        decompose_result.plot()
        plt.show()

        # Plot histogram of residuals
        residual.hist(bins=min(len(residual) // 10, 50), figsize=(8, 6), alpha=0.70)
        plt.title(f"Histogram of Residuals for {feature}")
        plt.show()

        # Plot autocorrelation
        plot_acf(residual,lags=lag)
        plt.title(f"Autocorrelation for {feature}")
        plt.show()
        
        # Perform ADF test
        dftest = adfuller(residual, autolag='AIC')

        if store:
            # Store results in a dictionary
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
            # Print ADF results
            print(f"\nADF Test Results for {feature}:")
            print("1. ADF Statistic:", dftest[0])
            print("2. P-Value:", dftest[1])
            print("3. Num of Lags:", dftest[2])
            print("4. Num of Observations:", dftest[3])
            print("5. Critical Values:")
            for key, val in dftest[4].items():
                print(f"\t{key}: {val}")

    except Exception as e:
        print(f"Error in decompose_adf: {e}")
        return None