import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# --- Configuration Constants ---
# Set the base directory to the root of the project 
# (assuming modeling.py is in 'src' and the project root is the parent)
BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')

# 🎯 CORRECTED PATH: Points to the final output of data_prep.py
PROCESSED_DATA_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'final_processed_data.csv') 

# --- Modeling Functions ---

def load_processed_data(file_path):
    """Loads the processed data from the specified path."""
    print(f"Attempting to load processed data from: {os.path.normpath(file_path)}")
    try:
        df = pd.read_csv(file_path)
        print(f"Data loaded successfully. Shape: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"Error: Processed data file not found at {os.path.normpath(file_path)}")
        return None
    except Exception as e:
        print(f"An error occurred during data loading: {e}")
        return None

def train_and_evaluate(df):
    """
    Splits data, trains a model, and evaluates its performance.
    
    NOTE: You must adjust the 'Target_Variable' and 'Feature_1', etc., 
    to match the actual column names in your processed data.
    """
    print("Starting model training and evaluation...")
    
    # --- Define Target and Features (ADJUST THESE COLUMN NAMES!) ---
    # Replace 'Target_Variable' with your actual target column (e.g., 'Claim_Occurrence')
    # Replace 'Feature_1', 'Feature_2', etc., with your actual feature columns
    
    try:
        # Assuming your target variable is named 'Target_Variable' for demonstration
        target_column = 'Target_Variable' 
        
        # Drop the target column to get features (X)
        X = df.drop(columns=[target_column])
        
        # Target variable (y)
        y = df[target_column]
        
    except KeyError as e:
        print(f"\n❌ ERROR: Column {e} not found in the DataFrame.")
        print("Please ensure you have run data_prep.py and adjust the column names in modeling.py.")
        return
        
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")
    
    # 1. Initialize and Train the Model (Using Logistic Regression as a placeholder)
    model = LogisticRegression(solver='liblinear', random_state=42)
    model.fit(X_train, y_train)
    
    # 2. Predict and Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print("-" * 30)
    print(f"Model Type: Logistic Regression")
    print(f"Test Set Accuracy: {accuracy:.4f}")
    print("-" * 30)
    
    # You would typically save the model here (e.g., using joblib)
    # print("Model training and evaluation complete.")
    
    return model

def main():
    """Main function to run the modeling workflow."""
    print("Starting machine learning modeling workflow...")
    
    # 1. Load Data
    data_df = load_processed_data(PROCESSED_DATA_PATH)
    
    if data_df is None:
        print("Modeling workflow terminated due to data loading error.")
        return
    
    # 2. Train and Evaluate
    trained_model = train_and_evaluate(data_df)
    
    # if trained_model:
    #     # Add logic here to save the model using DVC or joblib
    #     pass

    print("Modeling workflow finished.")

if __name__ == "__main__":
    main()