from predict import predict
from load import load_data, load_predictions_data
import os
from dotenv import load_dotenv
load_dotenv()

db_config = {
        "host": os.getenv("DB_HOST", "localhost"),
        "port": os.getenv("DB_PORT", "5432"),
        "user": os.getenv("DB_USER", "postgres"),
        "password": os.getenv("DB_PASSWORD", ""),  
        "dbname": os.getenv("DB_NAME", "Finance")
    }

def load_predicts():
    prediction_df = predict()

    load_predictions_data(prediction_df, db_config)


