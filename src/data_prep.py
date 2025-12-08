import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

# --- CONFIGURATION ---
TARGET = 'totalclaims' # Corrected from 'claim_amount' or 'claim_count' to use 'totalclaims'

# List of all features to be used in the model.
# NOTE: Using standardized (lower_snake_case) names here.
FEATURES = [
    'transactionmonth',
    'isvatregistered',
    'citizenship',
    'legaltype',
    'title',
    'language',
    'bank',
    'accounttype',
    'maritalstatus',
    'gender',
    'country',
    'province',
    'postalcode',
    'maincrestazone',
    'subcrestazone',
    'itemtype',
    'mmcode',
    'vehicletype',
    'registrationyear',
    'make',
    'model',
    'cylinders',
    'cubiccapacity',
    'kilowatts',
    'bodytype',
    'numberofdoors',
    'vehicleintrodate',
    'customvalueestimate',
    'alarmimmobiliser',
    'trackingdevice',
    'capitaloutstanding',
    'newvehicle',
    'writtenoff',
    'rebuilt',
    'converted',
    'crossborder',
    'numberofvehiclesinfleet', # Using standardized name
    'suminsured',
    'termfrequency',
    'calculatedpremiumperterm',
    'excessselected',
    'covercategory', # Will be one-hot encoded later
    'covertype',
    'covergroup',
    'section',
    'product',
    'statutoryclass',
    'statutoryrisktype',
    'totalpremium'
]

def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans, transforms, and prepares the raw data for model training.
    """
    print("Starting data preprocessing...")

    # --- 1. Standardize Column Names ---
    # Converts 'TotalClaims' to 'totalclaims', 'Cover Category' to 'cover_category', etc.
    data.columns = [col.lower().replace(' ', '_') for col in data.columns]
    
    # Optional: Print to verify standardized names for debugging
    # print("Standardized Columns:", data.columns.tolist())

    # --- 2. Feature Engineering (Placeholder/Example) ---
    # Create an 'exposure' feature (assuming 'TransactionMonth' is suitable for time calculation)
    # NOTE: This requires proper date parsing which is not shown but is crucial.
    # We will skip complex feature engineering for simplicity in this fix.

    # --- 3. Missing Value Imputation ---
    
    # Fix the KeyError for claim amount by using the correct standardized column name: 'totalclaims'
    if TARGET in data.columns:
        data[TARGET] = data[TARGET].fillna(0)
    
    # Example: Impute missing numerical features with the median
    for feature in ['kilowatts', 'suminsured']: # Use standardized names
        if feature in data.columns:
            data[feature].fillna(data[feature].median(), inplace=True)
            
    # Example: Impute missing categorical features with 'Missing'
    for feature in ['gender', 'maritalstatus']: # Use standardized names
        if feature in data.columns:
            data[feature].fillna('Missing', inplace=True)


    # --- 4. Encoding Categorical Variables ---
    
    # A. Label Encoding for high-cardinality/ordinal features
    # Example: Encoding 'covertype'
    if 'covertype' in data.columns:
        le = LabelEncoder()
        data['encoded_covertype'] = le.fit_transform(data['covertype'].astype(str))
        # Add the new encoded feature to the FEATURES list if you intend to use it
        # You'll need to remove the original 'covertype' from FEATURES later if you don't want both
    
    # B. One-Hot Encoding for lower-cardinality features
    # Example: One-Hot Encoding 'covercategory'
    if 'covercategory' in data.columns:
        data = pd.get_dummies(data, columns=['covercategory'], prefix='cat', dummy_na=False)
        # NOTE: When using get_dummies, you must update the FEATURES list after this step
        
    print("Data preprocessing complete.")
    return data

# --- EOF ---