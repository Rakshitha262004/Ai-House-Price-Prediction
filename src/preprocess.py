# ============================================
# PREPROCESS MODULE
# HOUSE PRICE PREDICTION
# ============================================

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import OrdinalEncoder

ENCODER_PATH = "models/encoder.pkl"
COLUMNS_PATH = "models/feature_columns.pkl"


def preprocess_data(df, fit=True):
    """
    Preprocess the dataframe.
    - fit=True  → used during training (fits + saves encoder)
    - fit=False → used during inference (loads saved encoder)
    """
    df = df.copy()

    # Drop irrelevant columns
    drop_cols = ["Id"]
    df = df.drop(columns=[col for col in drop_cols if col in df.columns])

    # Separate target if present
    target = None
    if "SalePrice" in df.columns:
        target = df["SalePrice"].copy()
        df = df.drop(columns=["SalePrice"])

    # Fill missing values
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].fillna("Missing")
        else:
            df[col] = df[col].fillna(df[col].median())

    # Encode categorical columns
    cat_cols = df.select_dtypes(include="object").columns.tolist()

    if fit:
        encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        df[cat_cols] = encoder.fit_transform(df[cat_cols])
        os.makedirs("models", exist_ok=True)
        joblib.dump(encoder, ENCODER_PATH)
        joblib.dump(df.columns.tolist(), COLUMNS_PATH)
        print(f"   Encoder saved → {ENCODER_PATH}")
        print(f"   Columns saved → {COLUMNS_PATH}")
    else:
        encoder = joblib.load(ENCODER_PATH)
        df[cat_cols] = encoder.transform(df[cat_cols])

    # Reattach target
    if target is not None:
        df["SalePrice"] = target.values

    return df