import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    mean_squared_error, r2_score, classification_report
)
import numpy as np

# Import the data preparation functions from the sibling file
# NOTE: This assumes data_prep.py exports the necessary preprocessed data objects
from data_prep import preprocess_data

# --- CONFIGURATION (Must match data_prep.py) ---
TARGET_CLAIM_AMOUNT = 'totalclaims'
TARGET_CLAIM_OCCURRED = 'claim_occurred'
SEED = 42

# Define a common set of features (using standardized names from your file)
# NOTE: Ensure these features are present AND cleaned/encoded in data_prep.py
COMMON_FEATURES = [
    'kilowatts',
    'suminsured',
    'totalpremium',
    'registrationyear',
    'numberofvehiclesinfleet',
    # Placeholder for a categorical feature you encoded (e.g., gender_man)
    'gender_man', 
    'province_quebec' # Placeholder for a one-hot encoded province
]

# --- Helper Function to get Split Datasets ---
def get_model_datasets(df_processed: pd.DataFrame):
    """
    Creates the four necessary train/test splits for the two models.
    """
    # 1. Feature Selection: Filter out non-feature columns
    feature_cols = [col for col in COMMON_FEATURES if col in df_processed.columns]
    
    # --- PROBABILITY MODEL DATA (Classification: Predict P(Claim)) ---
    df_prob = df_processed.copy()
    df_prob[TARGET_CLAIM_OCCURRED] = (df_prob[TARGET_CLAIM_AMOUNT] > 0).astype(int)
    
    X_prob = df_prob[feature_cols]
    y_prob = df_prob[TARGET_CLAIM_OCCURRED]
    
    # Split Probability data (Stratified split is important for classification)
    prob_splits = train_test_split(
        X_prob, y_prob, test_size=0.2, random_state=SEED, stratify=y_prob
    )
    
    # --- SEVERITY MODEL DATA (Regression: Predict Claim Amount | Claim Occurred) ---
    # Filter for policies where a claim occurred
    df_sev = df_processed[df_processed[TARGET_CLAIM_AMOUNT] > 0].copy()
    
    X_sev = df_sev[feature_cols]
    y_sev = df_sev[TARGET_CLAIM_AMOUNT]
    
    # Split Severity data
    sev_splits = train_test_split(
        X_sev, y_sev, test_size=0.2, random_state=SEED
    )
    
    return {
        'probability': prob_splits,
        'severity': sev_splits
    }

# --- 1. Probability Model (Classification) ---

def train_and_evaluate_probability_model(X_train, X_test, y_train, y_test):
    """
    Trains and evaluates a Classification model (Random Forest) for P(Claim).
    """
    print("\n--- Training Claim Probability Model (Random Forest Classifier) ---")
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=SEED, n_jobs=-1)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    print("\nClassification Report (Test Data):")
    print(classification_report(y_test, y_pred))

    return model

# --- 2. Severity Model (Regression) ---

def train_and_evaluate_severity_model(X_train, X_test, y_train, y_test):
    """
    Trains and evaluates a Regression model (Linear Regression) for Claim Severity.
    """
    print("\n--- Training Claim Severity Model (Linear Regression) ---")
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_pred[y_pred < 0] = 0 # Claims must be non-negative
    
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print("\nRegression Metrics (Test Data):")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    print(f"R-squared (R2) Score: {r2:.4f}")
    
    return model

# --- Main Execution Block ---

if __name__ == '__main__':
    # 1. Load and preprocess the data
    df_encoded = preprocess_data(pd.read_csv('data/MachineLearningRating_v3.txt', sep='\t'))
    
    # 2. Get the required splits
    datasets = get_model_datasets(df_encoded)
    
    # 3. Train and Evaluate Probability Model (Task 4 Classification)
    X_train_prob, X_test_prob, y_train_prob, y_test_prob = datasets['probability']
    prob_model = train_and_evaluate_probability_model(
        X_train_prob, X_test_prob, y_train_prob, y_test_prob
    )
    
    # 4. Train and Evaluate Severity Model (Task 4 Regression)
    X_train_sev, X_test_sev, y_train_sev, y_test_sev = datasets['severity']
    sev_model = train_and_evaluate_severity_model(
        X_train_sev, X_test_sev, y_train_sev, y_test_sev
    )
    
    # 5. Next steps: Saving models and Feature Importance (SHAP/LIME)
    print("\n✅ Task 4 Modeling Baseline Complete. Ready for Model Comparison and Interpretability.")