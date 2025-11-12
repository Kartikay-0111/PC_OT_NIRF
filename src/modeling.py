# src/modeling.py

"""
Fits statistical/ML models to quantify how subcomponents influence TLR score.
This directly implements "Phase 3: Machine Learning and Predictive Modelling".

Models:
1. Linear Regression: To find simple coefficients.
2. Random Forest Regressor: To find non-linear feature importances.

Run from project root:
    python src/modeling.py
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
import joblib

# --- Constants ---

# Assumes this script is in 'src/' and data is in 'data/processed/'
# UPDATE: Read the processed features file, not the raw input.
DATA_PATH = os.path.join(
    os.path.dirname(__file__), 
    "..", "data", "processed", "tlr_model_features.csv"
)

# Directory to save the trained model
MODEL_DIR = os.path.join(
    os.path.dirname(__file__), 
    "..", "models"
)
MODEL_PATH = os.path.join(MODEL_DIR, "tlr_rf_model.joblib")

# UPDATE: Use the original scores AND the new engineered features
FEATURES = [
    'ss_score', 
    'fsr_score', 
    'fqe_score', 
    'fru_score', 
    'oe_score', 
    'mir_score',
    'phd_per_faculty_ratio',
    'students_per_faculty',
    'total_expense_per_student'
]
TARGET = 'tlr_score'


def load_data(path):
    """Loads the processed TLR dataset."""
    if not os.path.exists(path):
        print(f"Error: Data file not found at {path}")
        print("Please run 'python src/preprocess.py' first to generate the data.")
        return None
    
    print(f"Loading data from {path}...")
    return pd.read_csv(path)

def print_metrics(model_name, y_true, y_pred):
    """Helper function to print model evaluation metrics."""
    r2 = r2_score(y_true, y_pred)
    # Calculate RMSE manually for backward compatibility with older sklearn versions
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    print(f"--- {model_name} Metrics ---")
    print(f"  R-squared (R²): {r2:.4f}")
    print(f"  Root Mean Squared Error (RMSE): {rmse:.4f}")
    print("-" * (len(model_name) + 20))

def main():
    """Main function to run the modeling pipeline."""
    # Ensure the output directory exists
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    df = load_data(DATA_PATH)
    if df is None:
        return
        
    print(f"Defining features (X): {FEATURES}")
    print(f"Defining target (y): {TARGET}")
    
    X = df[FEATURES]
    y = df[TARGET]
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"Training with {len(X_train)} samples, testing with {len(X_test)} samples.")

    # --- 1. Linear Regression ---
    print("\nTraining Linear Regression model...")
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    
    # Evaluate
    y_pred_lr = lr_model.predict(X_test)
    print_metrics("Linear Regression", y_test, y_pred_lr)
    
    # Show coefficients
    lr_coeffs = pd.DataFrame(
        lr_model.coef_,
        index=FEATURES,
        columns=['Coefficient']
    ).sort_values(by='Coefficient', ascending=False)
    print("Linear Regression Coefficients (Feature Importance):")
    print(lr_coeffs)
    print("\n")

    # --- 2. Random Forest Regressor ---
    print("Training Random Forest Regressor model...")
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42, oob_score=True)
    rf_model.fit(X_train, y_train)
    
    # Evaluate
    y_pred_rf = rf_model.predict(X_test)
    print_metrics("Random Forest", y_test, y_pred_rf)
    print(f"  Out-of-Bag (OOB) Score: {rf_model.oob_score_:.4f}")
    
    # Show feature importances
    rf_importances = pd.DataFrame(
        rf_model.feature_importances_,
        index=FEATURES,
        columns=['Importance']
    ).sort_values(by='Importance', ascending=False)
    print("Random Forest Feature Importances:")
    print(rf_importances)
    print("\n")

    # --- 3. Save the best model ---
    print(f"Saving Random Forest model to {MODEL_PATH}...")
    joblib.dump(rf_model, MODEL_PATH)
    
    print("✅ Modeling script finished.")


if __name__ == "__main__":
    main()