from process import process_data, split_dataframe_by_period, read_data, add_nasdaq_return
from backtest import backtest_top3_with_sharpe
import os 
import joblib
import xgboost as xgb
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt
import matplotlib.pyplot as plt

def predict():
    # Define the path to the models directory
    models_directory = "./models"
    
    # Load models and features
    model_types = ["lightgbm", "xgboost", "catboost"]  # Add or update model types as needed
    models = {}
    features = {}

    for model_type in model_types:
        # Load ROI models and features
        roi_model_path = os.path.join(models_directory, f"{model_type}_roi_model.pkl")
        roi_features_path = os.path.join(models_directory, f"{model_type}_roi_features.pkl")

        if os.path.exists(roi_model_path) and os.path.exists(roi_features_path):
            models[f"{model_type}_roi"] = joblib.load(roi_model_path)
            features[f"{model_type}_roi"] = joblib.load(roi_features_path)
        
        # Load Sharpe models and features
        sharpe_model_path = os.path.join(models_directory, f"{model_type}_sharpe_model.pkl")
        sharpe_features_path = os.path.join(models_directory, f"{model_type}_sharpe_features.pkl")

        if os.path.exists(sharpe_model_path) and os.path.exists(sharpe_features_path):
            models[f"{model_type}_sharpe"] = joblib.load(sharpe_model_path)
            features[f"{model_type}_sharpe"] = joblib.load(sharpe_features_path)
    
    # Print loaded models and features for verification
    print("Loaded models:", models.keys())
    print("Loaded features:", features.keys())

    # Read and process the data
    df = read_data("historical_stock_data")
    print("\nData read successfully.")

    train_df, test_df, validation_df, prediction_df = process_data(df, split_periods=True)
    print("Data processed successfully.")
    print(f"\nTrain shape: {train_df.shape}, Test shape: {test_df.shape}\n")

    unique_stocks = train_df['stock_name'].unique().tolist()
    print("Unique stocks:", unique_stocks)

    # Initialize stacking dataset
    stacking_train = pd.DataFrame()
    stacking_test = pd.DataFrame()
    stacking_val = pd.DataFrame()
    stacking_pred = pd.DataFrame()

    for model_key, model in models.items():
        feature_set = features.get(model_key, [])
        print(f"Running predictions for {model_key}")
        if feature_set:
            # Prepare training and testing data
            X_train = train_df[feature_set]
            X_test = test_df[feature_set]
            X_val = validation_df[feature_set]
            X_pred = prediction_df[feature_set]
            
            # Convert to DMatrix if the model is XGBoost
            if "xgboost" in model_key:
                X_train_dmatrix = xgb.DMatrix(X_train)
                X_test_dmatrix = xgb.DMatrix(X_test)
                X_val_dmatrix = xgb.DMatrix(X_val)
                X_pred_dmatrix = xgb.DMatrix(X_pred)
                train_predictions = model.predict(X_train_dmatrix)
                test_predictions = model.predict(X_test_dmatrix)
                val_predictions = model.predict(X_val_dmatrix)
                pred_predictions = model.predict(X_pred_dmatrix)
            else:
                train_predictions = model.predict(X_train)
                test_predictions = model.predict(X_test)
                val_predictions = model.predict(X_val)
                pred_predictions = model.predict(X_pred)
            
            # Add predictions as features for stacking
            stacking_train[model_key] = train_predictions
            stacking_test[model_key] = test_predictions
            stacking_val[model_key] = val_predictions
            stacking_pred[model_key] = pred_predictions

    # Train the stacking model
    y_train = train_df['next_weekly_return']
    y_test = test_df['next_weekly_return']
    y_val = validation_df['next_weekly_return']
    y_pred = prediction_df['next_weekly_return']

    print("\nTraining stacking model...")
    stacking_model = RandomForestRegressor(n_estimators=100, random_state=42)
    stacking_model.fit(stacking_train, y_train)

    # Evaluate the stacking model
    stacking_predictions = stacking_model.predict(stacking_val)
    rmse = sqrt(mean_squared_error(y_val, stacking_predictions))
    print("\nStacking Model Metrics(Valid data 2023-2024):")
    print(f"\nStacking Model RMSE: {rmse:.4f}")
    mae = mean_absolute_error(y_val, stacking_predictions)
    print(f"Stacking Model MAE: {mae:.4f}")
    # Backtesting using stacking model predictions
    predictions_df = pd.DataFrame({
        'date': validation_df['date'],
        'stock_name': validation_df['stock_name'],
        'predicted_return': stacking_predictions
    })

    final_portfolio_value, net_profit, roi, sharpe_ratio, actions_df = backtest_top3_with_sharpe(
        predictions_df, df, unique_stocks
    )

    print(f"  ROI: {roi:.2f}%")
    print(f"  Sharpe Ratio: {sharpe_ratio:.4f}")
    print(f"  Net Profit: {net_profit:.2f}")
    print(f"  Final Portfolio Value : {final_portfolio_value:.2f}")
    print(f" Actions :\n {actions_df.head()}")
        # Predict for prediction_df using the stacking model
    stacking_predictions = stacking_model.predict(stacking_pred)

    rmse = sqrt(mean_squared_error(y_pred, stacking_predictions))
    print("\nStacking Model Metrics(Pred data 2024):")
    print(f"\nStacking Model RMSE: {rmse:.4f}")
    mae = mean_absolute_error(y_pred, stacking_predictions)
    print(f"Stacking Model MAE: {mae:.4f}")

    # Backtesting using stacking model predictions
    predictions_df = pd.DataFrame({
        'date': prediction_df['date'],
        'stock_name': prediction_df['stock_name'],
        'predicted_return': stacking_predictions,
        'return' : prediction_df['weekly_return'],
        'open' : prediction_df['open'],
        'adj_close' : prediction_df['adj_close'] 
    })

    final_portfolio_value, net_profit, roi, sharpe_ratio, actions_df = backtest_top3_with_sharpe(
        predictions_df, df, unique_stocks
    )

    print(f"  ROI: {roi:.2f}%")
    print(f"  Sharpe Ratio: {sharpe_ratio:.4f}")
    print(f"  Net Profit: {net_profit:.2f}")
    print(f"  Final Portfolio Value : {final_portfolio_value:.2f}\n")
    print(f"  Benchmark ROI (^IXIC) : 28,64%")
    print(f"\nAction DataFrame(2024):\n {actions_df.head()}")
    print(f"\nAction DataFrame(2024):\n {actions_df.sort_values(by='profit_percentage', ascending=False).head()}")


    # Plotting
    plt.figure(figsize=(14, 7))
    plt.plot(prediction_df['date'], prediction_df['weekly_return'], label='Return (Weekly)')
    plt.plot(predictions_df['date'], predictions_df['predicted_return'], label='Predicted Return')
    plt.xlabel('Date')
    plt.ylabel('Return')
    plt.title('Weekly Return vs Predicted Return')
    plt.legend()
    plt.grid()
    plt.show()




    return predictions_df

