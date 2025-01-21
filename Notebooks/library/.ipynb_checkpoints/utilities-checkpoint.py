import pandas as pd

def check_missing_or_unavailable(df, unavailable_value="Value not available"):
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