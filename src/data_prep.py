import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# --- Configuration Constants ---
BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
PROCESSED_DATA_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'final_processed_data.csv') 

# --- Modeling Functions ---

def load_processed_data(file_path):
    """Loads the processed data from the specified path."""
    print(f"Attempting to load processed data from: {os.path.normpath(file_path)}")
    try:
        df = pd.read_csv(file_path)
        print(f"Data loaded successfully. Shape: {df.shape}")
        # Display the columns to help with debugging the next step
        print(f"Columns: {list(df.columns)}") 
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
    """
    print("Starting model training and evaluation...")
    
    # 🎯 CRITICAL CORRECTION: 
    # REPLACE 'YOUR_ACTUAL_TARGET_COLUMN_NAME' with the correct column name from your CSV file.
    # Example: If your target is named 'Is_Claim', use that name.
    target_column = 'YOUR_ACTUAL_TARGET_COLUMN_NAME'  
    
    try:
        # X: Features (all columns EXCEPT the target)
        X = df.drop(columns=[target_column])
        
        # y: Target variable
        y = df[target_column]
        
    except KeyError as e:
        print(f"\n❌ ERROR: Column {e} not found in the DataFrame.")
        print(f"Please replace the placeholder '{target_column}' with the correct target column name.")
        print(f"Available columns are: {list(df.columns)}")
        return None
        
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")
    
    # Initialize and Train the Model
    model = LogisticRegression(solver='liblinear', random_state=42, max_iter=1000) # Increased max_iter for safety
    model.fit(X_train, y_train)
    
    # Predict and Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print("-" * 30)
    print(f"Model Type: Logistic Regression")
    print(f"Test Set Accuracy: {accuracy:.4f}")
    print("-" * 30)
    
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
    
    print("Modeling workflow finished.")

if __name__ == "__main__":
    main()