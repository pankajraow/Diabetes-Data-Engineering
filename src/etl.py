import pandas as pd
import pandera as pa
from pandera import Column, Check
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
import os

# --- Configuration ---
DATA_PATH = os.path.join("data", "diabetes_prediction_dataset.csv")
PROCESSED_DIR = os.path.join("data", "processed")
OUT_DIR = os.path.join("output")

# --- Schema Definition ---
# Defining strict schema for validation
schema = pa.DataFrameSchema({
    "gender": Column(str, Check.isin(["Male", "Female", "Other"])),
    "age": Column(float, Check.ge(0)),
    "hypertension": Column(int, Check.isin([0, 1])),
    "heart_disease": Column(int, Check.isin([0, 1])),
    "smoking_history": Column(str, Check.isin(
        ["never", "current", "former", "ever", "not current", "No Info"])),
    "bmi": Column(float, Check.gt(0)),
    "HbA1c_level": Column(float, Check.gt(0)),
    "blood_glucose_level": Column(float, Check.gt(0)),
    "diabetes": Column(int, Check.isin([0, 1])),
})

def load_data(filepath):
    print(f"Loading data from {filepath}...")
    try:
        df = pd.read_csv(filepath)
        print(f"Loaded {len(df)} rows.")
        return df
    except FileNotFoundError:
        print(f"Error: File {filepath} not found.")
        return None

def validate_and_clean(df):
    print("Validating schema...")
    try:
        df = schema.validate(df, lazy=True)
        print("Schema validation passed.")
    except pa.errors.SchemaErrors as err:
        print("Schema validation failed:")
        print(err.failure_cases)
        # In a real pipeline, we might drop bad rows or quarantine them.
        # For this challenge, we'll try to drop rows that don't match strict types if possible,
        # but pandera raises error. We will proceed if only minor issues, or stop.
        # For simplicity in this script, we'll assume we want to clean 'bad' data.
        # Let's quarantine invalid rows based on logic manually if needed.
    
    # Cleaning
    initial_count = len(df)
    
    # 1. Duplicates
    df = df.drop_duplicates()
    print(f"Dropped {initial_count - len(df)} duplicate rows.")
    
    # 2. Missing values - For this dataset, usually no missing in standard version, 
    # but let's check.
    if df.isnull().sum().sum() > 0:
        print("Missing values detected. Dropping...")
        df = df.dropna()
    
    return df

def process_data(df):
    print("Processing data...")
    
    # Features and Target
    X = df.drop("diabetes", axis=1)
    y = df["diabetes"]
    
    # numeric and categorical columns
    numeric_cols = ["age", "bmi", "HbA1c_level", "blood_glucose_level"]
    cat_cols = ["gender", "smoking_history"]
    passthrough_cols = ["hypertension", "heart_disease"]
    
    # Encoding Categorical
    # using get_dummies for simplicity and readability in output df
    X_encoded = pd.get_dummies(X, columns=cat_cols, drop_first=True)
    
    # Scaling Numeric
    scaler = StandardScaler()
    X_encoded[numeric_cols] = scaler.fit_transform(X_encoded[numeric_cols])
    
    # Combine back (y is separated, but usually for splits we keep X and y together or sync)
    # Let's return X_encoded and y
    return X_encoded, y

def split_and_save(X, y):
    print("Splitting data...")
    # Stratified split: Train (70%), Test (15%), Val (15%)
    # First split: Train (70%) and Temp (30%)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )
    # Second split: Test (15% orig -> 50% of Temp) and Val
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
    )
    
    print(f"Train size: {len(X_train)}")
    print(f"Val size:   {len(X_val)}")
    print(f"Test size:  {len(X_test)}")
    
    # Save
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    
    # Joining X and y for saving (often easier for loading later)
    train_df = pd.concat([X_train, y_train], axis=1)
    val_df = pd.concat([X_val, y_val], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)
    
    train_df.to_csv(os.path.join(PROCESSED_DIR, "train.csv"), index=False)
    val_df.to_csv(os.path.join(PROCESSED_DIR, "val.csv"), index=False)
    test_df.to_csv(os.path.join(PROCESSED_DIR, "test.csv"), index=False)
    print(f"Saved processed datasets to {PROCESSED_DIR}")

def main():
    df = load_data(DATA_PATH)
    if df is not None:
        df = validate_and_clean(df)
        X, y = process_data(df)
        split_and_save(X, y)

if __name__ == "__main__":
    main()
