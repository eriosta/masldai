import pickle
import xgboost as xgb
import numpy as np
np.random.seed(42)

# FibroX AI-specific features and logic
MODEL_PATH = "fibrox.pkl"
MODEL_FEATURES = [
    'Age_in_years_at_screening',
    'Glycohemoglobin (%)',
    'Alanine Aminotransferase (ALT) (U/L)',
    'Aspartate Aminotransferase (AST) (U/L)',
    'Platelet count (1000 cells/uL)',
    'Body Mass Index (kg/m**2)',
    'GFR'
]

def load_model():
    with open(MODEL_PATH, 'rb') as file:
        return pickle.load(file)

def calculate_gfr(serum_cr, age, is_female):
    gfr = 175 * (serum_cr ** -1.154) * (age ** -0.203)
    if is_female:
        gfr *= 0.742
    return gfr

def predict(model, data):
    threshold = 0.42690  # Custom threshold
    data_dmatrix = xgb.DMatrix(data[MODEL_FEATURES])
    y_pred_proba = model.predict(data_dmatrix)
    y_pred = (y_pred_proba >= threshold).astype(int)
    return y_pred, y_pred_proba
