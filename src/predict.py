# ============================================
# PREDICT MODULE
# HOUSE PRICE PREDICTION
# ============================================

import joblib
import numpy as np
import pandas as pd

MODEL_PATH = "models/model.pkl"
COLUMNS_PATH = "models/feature_columns.pkl"


def load_artifacts():
    model = joblib.load(MODEL_PATH)
    columns = joblib.load(COLUMNS_PATH)
    return model, columns


def predict(feature_dict: dict) -> float:
    """
    Predict house price from a dictionary of feature values.

    Args:
        feature_dict: dict mapping column names → values
                      (only the features you know; rest default to 0)

    Returns:
        Predicted price as a float
    """
    model, columns = load_artifacts()

    # Build a zero-filled dataframe with the correct column order
    input_df = pd.DataFrame([np.zeros(len(columns))], columns=columns)

    # Fill in the provided feature values
    for col, val in feature_dict.items():
        if col in input_df.columns:
            input_df[col] = val

    prediction = model.predict(input_df)[0]
    return float(prediction)