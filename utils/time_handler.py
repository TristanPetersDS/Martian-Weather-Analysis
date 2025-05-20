import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from datetime import datetime, timedelta


# ─────────────────────────────────────────────────────────────────────────────
# Time Conversion Utilities
# ─────────────────────────────────────────────────────────────────────────────

def time_to_minutes(t):
    """
    Convert a time object to total minutes since midnight.

    Parameters:
        t (datetime.time): Time object to convert.

    Returns:
        float: Minutes since midnight, or NaN if input is missing.
    """
    if pd.isnull(t):
        return np.nan
    return t.hour * 60 + t.minute


def minutes_to_time(m):
    """
    Convert total minutes since midnight to a time object.

    Parameters:
        m (float): Minutes since midnight.

    Returns:
        datetime.time: Time object, or NaN if input is missing.
    """
    if pd.isnull(m):
        return np.nan
    return (datetime.min + timedelta(minutes=m)).time()


def time_to_seconds(t):
    """
    Convert a time object to total seconds since midnight.

    Parameters:
        t (datetime.time): Time object to convert.

    Returns:
        float: Seconds since midnight, or NaN if input is missing.
    """
    if pd.isnull(t):
        return np.nan
    return t.hour * 3600 + t.minute * 60 + t.second


def seconds_to_time(s):
    """
    Convert total seconds since midnight to a time object.

    Parameters:
        s (float): Seconds since midnight.

    Returns:
        datetime.time: Time object, or NaN if input is missing.
    """
    if pd.isnull(s):
        return np.nan
    total_seconds = int(round(s))
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    return datetime.min.replace(hour=hours, minute=minutes, second=seconds).time()
