from pyspark.sql import SparkSession
from extract import extract_data
from load import load_data
from transformation import transform_data
import os
from dotenv import load_dotenv
from load_predicts import load_predicts
# Load environment variables from .env file
load_dotenv()

def main():
    # Initialize Spark session
    spark = SparkSession.builder \
        .appName("Yahoo Finance to PostgreSQL") \
        .getOrCreate()

    # Define the stock index and additional tickers
    index_list = ['AAPL', 'NVDA', 'MSFT', 'AVGO', 'META', 'AMZN', 'TSLA', '^GSPC', 'NQ=F', '^IXIC']

    # Step 1: Extract data
    print("Extracting data...")
    raw_data = extract_data(index_list)
    if not raw_data:
        print("No data extracted. Exiting.")
        return

    # Step 2: Transform data
    print("\nTransforming data...\n")
    transformed_data = transform_data(raw_data, spark)  # Pass the Spark session
    if not transformed_data:
        print("No data transformed. Exiting.")
        return

    # Define database configuration using environment variables
    db_config = {
        "host": os.getenv("DB_HOST", "localhost"),
        "port": os.getenv("DB_PORT", "5432"),
        "user": os.getenv("DB_USER", "postgres"),
        "password": os.getenv("DB_PASSWORD", ""),  
        "dbname": os.getenv("DB_NAME", "Finance")
    }
    

    # Table name for loading data
    table_name = "historical_stock_data"

    # Step 3: Load data
    print("Loading data into PostgreSQL...")
    load_data(transformed_data, table_name, db_config)  # Pass the Spark session
    print("ETL pipeline completed successfully.")

    
    print(f"Prediction Process starting..")
    print(f"Reading data from PostgreSQL..")
    load_predicts()

    # Stop the Spark session
    spark.stop()

if __name__ == '__main__':
    main()