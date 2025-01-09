from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, FloatType, DateType, LongType
import pandas as pd

def transform_data(raw_data, spark):
    """
    Transform raw data into a Spark DataFrame.
    """

    # Define schema for the DataFrame
    schema = StructType([
        StructField("Date", DateType(), True),
        StructField("Adj Close", FloatType(), True),
        StructField("Close", FloatType(), True),
        StructField("High", FloatType(), True),
        StructField("Low", FloatType(), True),
        StructField("Open", FloatType(), True),
        StructField("Volume", LongType(), True),
        StructField("Stock_Name", StringType(), True)  # Added Stock_Name field
    ])

    transformed_data = []

    for ticker, data in raw_data.items():
        try:
            print(f"Transforming data for {ticker}...")

            # Reset index to convert 'Date' from index to column
            data = data.reset_index()

            # Debugging: Print the columns of the DataFrame
            print(f"Columns in data for {ticker}: {data.columns.tolist()}")

            # Check if 'Date' column exists
            if 'Date' not in data.columns:
                print(f"Error: 'Date' column not found for {ticker}.")
                continue

            # Check for null values in the 'Date' column
            if data['Date'].isnull().any():
                print(f"Warning: Null values found in 'Date' column for {ticker}.")
                data = data.dropna(subset=['Date'])

            # Convert the 'Date' column to datetime and then to date
            data['Date'] = pd.to_datetime(data['Date'], errors='coerce').dt.date  # Convert to date

            # Check for invalid dates after conversion
            if data['Date'].isnull().any():
                print(f"Warning: Invalid date values found for {ticker} after conversion.")
                data = data.dropna(subset=['Date'])

            # Add the ticker as a new column
            data['Stock_Name'] = ticker  # Add stock name

            # Convert Pandas DataFrame to Spark DataFrame
            spark_df = spark.createDataFrame(data, schema=schema)

            # Collect data back to a list of rows
            transformed_data.extend(spark_df.collect())

            print(f"Data for {ticker} transformed successfully.")
        except Exception as e:
            print(f"Error transforming data for {ticker}: {e}")

    spark.stop()
    return transformed_data