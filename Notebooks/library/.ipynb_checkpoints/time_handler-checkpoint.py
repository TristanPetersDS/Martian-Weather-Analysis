import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from datetime import datetime, timedelta

# Convert sunrise times to total minutes since midnight
def time_to_minutes(t):
    if pd.isnull(t):
        return np.nan
    return t.hour * 60 + t.minute

# Convert total minutes back to time
def minutes_to_time(m):
    if pd.isnull(m):
        return np.nan
    return (datetime.min + timedelta(minutes=m)).time()

# Convert sunrise times to total seconds since midnight
def time_to_seconds(t):
    if pd.isnull(t):
        return np.nan
    return t.hour * 3600 + t.minute * 60 + t.second

# Convert total seconds back to time, rounded to the nearest second
def seconds_to_time(s):
    if pd.isnull(s):
        return np.nan
    total_seconds = int(round(s))  # Round to the nearest second
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    return datetime.min.replace(hour=hours, minute=minutes, second=seconds).time()
