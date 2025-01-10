# Import libraries
import os
import psycopg2
import pandas as pd
from dotenv import load_dotenv
import numpy as np
import warnings
warnings.filterwarnings('ignore')

load_dotenv()



def read_data(table_name):
    """
    Read data from the specified PostgreSQL table and return it as a DataFrame.
    """
    # Database configuration from environment variables
    db_config = {
        "host": os.getenv("DB_HOST"),
        "port": os.getenv("DB_PORT"),
        "user": os.getenv("DB_USER"),
        "password": os.getenv("DB_PASSWORD"),
        "dbname": os.getenv("DB_NAME")
    }

    try:
        # Connect to the PostgreSQL database
        conn = psycopg2.connect(**db_config)
        
        # Create a cursor object
        cursor = conn.cursor()
        
        # Execute a query to fetch data
        query = f"SELECT * FROM {table_name};"
        cursor.execute(query)
        
        # Fetch all rows from the executed query
        rows = cursor.fetchall()
        
        # Convert the data to a Pandas DataFrame
        columns = [desc[0] for desc in cursor.description]  # Get column names
        df = pd.DataFrame(rows, columns=columns)
        
        # Convert the 'date' column to datetime
        df['date'] = pd.to_datetime(df['date'], errors='coerce')  # Convert to datetime
        
        return df  # Return the DataFrame
        
    except Exception as e:
        print(f"Error reading data from PostgreSQL: {e}")
        return None  # Return None in case of an error
    finally:
        if conn:
            cursor.close()
            conn.close()


# Function to add NASDAQ return
def add_nasdaq_return(df):
    """Add NASDAQ return to the dataset"""
    nasdaq_data = df[df['stock_name'] == '^IXIC'][['date', 'adj_close']]
    nasdaq_data['nasdaq_return'] = nasdaq_data['adj_close'].pct_change() * 100
    nasdaq_data['date'] = pd.to_datetime(nasdaq_data['date'])
    nasdaq_data = nasdaq_data[['date', 'nasdaq_return']]

    # Merge NASDAQ return with other stocks
    df['date'] = pd.to_datetime(df['date'])
    data = df[df['stock_name'] != '^IXIC']
    df = pd.merge(data, nasdaq_data, on='date', how='left')
    return df

def resample_stock(df):
    # Ensure Date column is a datetime object
    if not isinstance(df.index, pd.DatetimeIndex):
        df.rename(columns={'Date': 'date'}, inplace=True)
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)

    # Define resampling function
    def resample_group(group):
        return group.resample('W-FRI').agg({
            'open': 'first',          # First value of the week
            'high': 'max',            # Highest value of the week
            'low': 'min',             # Lowest value of the week
            'close': 'last',          # Last value of the week
            'adj_close': 'last',      # Adjusted close for the week
            'volume': 'sum',          # Total volume of the week
        })
    # Apply resampling to each stock group
    resampled_df = df.groupby('stock_name').apply(resample_group)
    resampled_df.rename(columns={'Date': 'date'}, inplace=True)
    # Reset index for grouped data
    resampled_df.reset_index(level=0, inplace=True)  # Reset 'stock_name'
    resampled_df.reset_index(inplace=True)          # Reset Date index

    # Sort by Date column
    resampled_df = resampled_df.sort_values(by='date')

    return resampled_df

def feature_engineering(df):
    """Add technical indicators, weekly lagged features, and engineered features to the DataFrame."""
    df = df.sort_values(['stock_name', 'date'])
    grouped = df.groupby('stock_name')

    # Core weekly features
    df['weekly_price_diff'] = grouped['adj_close'].transform('last') - grouped['open'].transform('first')
    df['range'] = grouped['high'].transform('max') - grouped['low'].transform('min')
    df['gap'] = grouped['open'].diff()
    df['volatility'] = df['range'] / grouped['open'].transform('first')

    # Volume indicators
    df['volume_ma'] = grouped['volume'].transform(lambda x: x.rolling(window=4, min_periods=1).mean())
    df['volume_ratio'] = df['volume'] / df['volume_ma']

    # RSI
    def calculate_rsi(series, period=2):
        delta = series.diff()
        gain = np.maximum(delta, 0)
        loss = np.abs(np.minimum(delta, 0))
        avg_gain = gain.rolling(window=period, min_periods=1).mean()
        avg_loss = loss.rolling(window=period, min_periods=1).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    df['rsi_2w'] = grouped['adj_close'].transform(lambda x: calculate_rsi(x, period=2))
    df['normalized_rsi_2w'] = (df['rsi_2w'] - 30) / (70 - 30)
    df['momentum_4w'] = grouped['adj_close'].diff(periods=4)
    df['momentum_8w'] = grouped['adj_close'].diff(periods=8)

    # MACD
    df['ema12'] = grouped['adj_close'].transform(lambda x: x.ewm(span=12, adjust=False).mean())
    df['ema26'] = grouped['adj_close'].transform(lambda x: x.ewm(span=26, adjust=False).mean())
    df['macd'] = df['ema12'] - df['ema26']
    df['signal_line'] = grouped['macd'].transform(lambda x: x.ewm(span=9, adjust=False).mean())
    df['MACD_Distance'] = df['macd'] - df['signal_line']
    df['macd_short'] = grouped['adj_close'].transform(lambda x: x.ewm(span=6, adjust=False).mean())
    df['macd_long'] = grouped['adj_close'].transform(lambda x: x.ewm(span=19, adjust=False).mean())
    df['macd_diff'] = df['macd_short'] - df['macd_long']


    # Stochastic oscillator
    stoch_window = 14
    df['low14'] = grouped['adj_close'].transform(lambda x: x.rolling(window=stoch_window).min())
    df['high14'] = grouped['adj_close'].transform(lambda x: x.rolling(window=stoch_window).max())
    df['%K'] = ((df['adj_close'] - df['low14']) / (df['high14'] - df['low14'])) * 100
    df['%D'] = grouped['%K'].transform(lambda x: x.rolling(window=3).mean())
    df['Stochastic_K_D_Distance'] = df['%K'] - df['%D']

    # OBV
    df['obv'] = grouped.apply(
        lambda g: (np.sign(g['adj_close'].diff()) * g['volume']).fillna(0).cumsum()
    ).reset_index(level=0, drop=True)

    # VWAP
    df['vwap'] = (df['adj_close'] * df['volume']).cumsum() / df['volume'].cumsum()
    df['VWAP_Distance'] = df['adj_close'] - df['vwap']

    # True Range (TR) calculation
    df['hl'] = df['high'] - df['low']
    df['hc'] = (df['high'] - grouped['adj_close'].shift()).abs()
    df['lc'] = (df['low'] - grouped['adj_close'].shift()).abs()
    df['tr'] = df[['hl', 'hc', 'lc']].max(axis=1)

    # Average True Range (ATR)
    df['atr'] = grouped['tr'].transform(lambda x: x.rolling(window=14).mean())
    df['ATR_Percentage'] = df['atr'] / df['adj_close'] * 100

    # Pivot Points
    df['Pivot'] = (df['high'] + df['low'] + df['adj_close']) / 3
    df['Pivot_Distance'] = df['adj_close'] - df['Pivot']
    df['Support_Distance'] = df['adj_close'] - (2 * df['Pivot'] - df['high'])
    df['Resistance_Distance'] = df['adj_close'] - (2 * df['Pivot'] - df['low'])

    # Weekly returns
    df['weekly_return'] = grouped['adj_close'].pct_change()
    df['next_weekly_return'] = grouped['weekly_return'].shift(-1)

    # Weekly lagged features
    lag_features = ['adj_close', 'volume', 'weekly_return']
    for lag in range(1, 5):  # Create weekly lag features for 1 to 4 weeks
        for feature in lag_features:
            df[f'{feature}_lag_{lag}w'] = grouped[feature].shift(lag)

    # Additional features
    df['price_to_range_ratio'] = df['adj_close'] / df['range']
    df['volume_change'] = grouped['volume'].pct_change()
    df['adj_close_change'] = grouped['adj_close'].pct_change()
    df['momentum_2w'] = grouped['adj_close'].diff(periods=2)
    for lag in range(5, 9):  # Adding lags for 5 to 8 weeks
        df[f'weekly_return_lag_{lag}w'] = grouped['weekly_return'].shift(lag)


    # Rolling stats
    df['rolling_mean_4w'] = grouped['adj_close'].transform(lambda x: x.rolling(window=4).mean())
    df['rolling_std_4w'] = grouped['adj_close'].transform(lambda x: x.rolling(window=4).std())
    df['bollinger_upper'] = df['rolling_mean_4w'] + (2 * df['rolling_std_4w'])
    df['bollinger_lower'] = df['rolling_mean_4w'] - (2 * df['rolling_std_4w'])
    df['bollinger_bandwidth'] = (df['bollinger_upper'] - df['bollinger_lower']) / df['rolling_mean_4w']
    df['rolling_skew_4w'] = grouped['weekly_return'].transform(lambda x: x.rolling(window=4).skew())
    df['rolling_kurt_4w'] = grouped['weekly_return'].transform(lambda x: x.rolling(window=4).kurt())
    df['sharpe_ratio'] = df['weekly_return'] / df['rolling_std_4w']

    df['day_of_week'] = df['date'].dt.dayofweek
    df['week_of_year'] = df['date'].dt.isocalendar().week
    df['month'] = df['date'].dt.month


    return df



def split_dataframe_by_period(df, split_periods=True):
    """
    Splits the DataFrame into Train, Test, Validation, and Prediction periods.

    Parameters:
        df (pd.DataFrame): The input DataFrame containing a 'date' column.

    Returns:
        tuple: Four DataFrames for Train, Test, Validation, and Prediction periods.
    """
    # Ensure the 'date' column is in datetime format
    df['date'] = pd.to_datetime(df['date'])
    if not split_periods:
        return df
    # Define date ranges for each period
    train_df = df[(df['date'] >= '2012-01-01') & (df['date'] < '2020-01-01')]
    test_df = df[(df['date'] >= '2020-01-01') & (df['date'] < '2023-01-01')]
    validation_df = df[(df['date'] >= '2023-01-01') & (df['date'] < '2024-01-01')]
    prediction_df = df[df['date'] >= '2024-01-01']

    return train_df, test_df, validation_df, prediction_df

def process_data(df, split_periods=False):
    """
    Main function to process the DataFrame and optionally split it into periods.

    Parameters:
        df (pd.DataFrame): The input DataFrame containing a 'date' column.
        split_periods (bool): Whether to split the DataFrame into Train, Test, Validation, and Prediction periods.

    Returns:
        pd.DataFrame or tuple: Processed DataFrame or (Train, Test, Validation, Prediction) DataFrames.
    """
    # Process the DataFrame
    df = add_nasdaq_return(df)
    df = resample_stock(df)
    df = feature_engineering(df)
    df = df.dropna()  # Drop NaN values if needed
    df = df.sort_values(by='date')
    # Split into periods if requested
    if split_periods:
        return split_dataframe_by_period(df)

    return df