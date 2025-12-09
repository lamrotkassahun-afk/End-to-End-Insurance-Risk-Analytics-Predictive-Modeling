import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Define file paths
PROCESSED_DATA_PATH = os.path.join('data', 'processed', 'final_processed_data.csv')
MODEL_OUTPUT_PATH = os.path.join('models', 'severity_model.joblib')

def load_processed_data(file_path):
    """
    Loads the final processed data from a specified CSV file.
    
    Args:
        file_path (str): The full path to the processed CSV file.
        
    Returns:
        pd.DataFrame: The loaded DataFrame, or None if an error occurs.
    """
    print(f"Attempting to load processed data from: {file_path}")
    try:
        # FIX 1: Add sep='|' to correctly parse the pipe-delimited file
        df = pd.read_csv(file_path, sep='|') 
        
        print(f"Data loaded successfully. Shape: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred during data loading: {e}")
        return None

def train_and_evaluate(df):
    """
    Trains a predictive model (Linear Regression) and evaluates its performance.
    
    Args:
        df (pd.DataFrame): The pre-processed DataFrame.
    """
    TARGET_COLUMN = 'TotalClaims'
    
    if TARGET_COLUMN not in df.columns:
        print(f"Error: Target column '{TARGET_COLUMN}' not found in DataFrame columns.")
        return

    # **FIX 3: Drop rows where the TARGET_COLUMN is NaN**
    # This prevents the ValueError: Input y contains NaN
    print(f"Initial row count: {len(df)}")
    df.dropna(subset=[TARGET_COLUMN], inplace=True)
    print(f"Row count after dropping target NaNs: {len(df)}")


    # 1. Feature and Target Split
    # Drop the target and unnecessary non-numeric ID columns
    X = df.drop(columns=[TARGET_COLUMN, 'UnderwrittenCoverID', 'PolicyID'], errors='ignore')
    y = df[TARGET_COLUMN]

    # Convert non-numeric columns to numeric (e.g., one-hot encoding or simply selecting numeric types)
    X = X.select_dtypes(include=np.number)
    X = X.fillna(0) # Fill any remaining NaNs in the features (X)
    
    # 2. Data Splitting
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Data split successfully. Training samples: {len(X_train)}, Testing samples: {len(X_test)}")

    # 3. Model Training (Linear Regression for insurance severity modeling)
    print("Training model...")
    model = LinearRegression()
    # model.fit will now succeed because y_train is guaranteed to be clean
    model.fit(X_train, y_train)
    print("Model training finished.")
    
    # 4. Prediction and Evaluation
    y_pred = model.predict(X_test)
    
    # Evaluation metrics for regression
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    print("\n--- Model Evaluation Results (Regression) ---")
    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    print(f"R-squared (R²): {r2:.4f}")
    print("------------------------------------------")

    # 5. Model Saving
    os.makedirs(os.path.dirname(MODEL_OUTPUT_PATH), exist_ok=True)
    joblib.dump(model, MODEL_OUTPUT_PATH)
    print(f"Model saved successfully to: {MODEL_OUTPUT_PATH}")

def run_modeling_workflow():
    """
    Main function to run the entire modeling workflow.
    """
    df = load_processed_data(PROCESSED_DATA_PATH)
    
    if df is not None and not df.empty:
        train_and_evaluate(df)
        
    print("Modeling workflow finished.")

if __name__ == "__main__":
    run_modeling_workflow()