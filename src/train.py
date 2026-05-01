# ============================================
# INDUSTRY LEVEL TRAINING SCRIPT
# HOUSE PRICE PREDICTION
# ============================================

import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from src.preprocess import preprocess_data

# --------------------------------------------
# CREATE REQUIRED FOLDERS
# --------------------------------------------
os.makedirs("models", exist_ok=True)
os.makedirs("images", exist_ok=True)
os.makedirs("outputs", exist_ok=True)


# --------------------------------------------
# EVALUATION FUNCTION
# --------------------------------------------
def evaluate_model(name, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    print(f"\n  {name} Performance:")
    print(f"   MAE  : {mae:,.2f}")
    print(f"   RMSE : {rmse:,.2f}")
    print(f"   R²   : {r2:.4f}")

    return mae, rmse, r2


# --------------------------------------------
# MAIN TRAIN FUNCTION
# --------------------------------------------
def train_model():
    print("Loading dataset...")
    df = pd.read_csv("data/train.csv")

    print("Preprocessing data...")
    # fit=True → fits encoder and saves column list
    df = preprocess_data(df, fit=True)

    # ----------------------------------------
    # EDA VISUALS
    # ----------------------------------------
    print("Generating EDA visuals...")

    numeric_df = df.select_dtypes(include="number")

    plt.figure(figsize=(12, 8))
    sns.heatmap(numeric_df.corr(), cmap="coolwarm", linewidths=0.5)
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig("images/heatmap.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    sns.histplot(df["SalePrice"], kde=True, color="steelblue")
    plt.title("Sale Price Distribution")
    plt.tight_layout()
    plt.savefig("images/price_distribution.png", dpi=150)
    plt.close()

    # ----------------------------------------
    # SPLIT DATA
    # ----------------------------------------
    X = df.drop("SalePrice", axis=1)
    y = df["SalePrice"]

    print("Splitting dataset...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ----------------------------------------
    # DEFINE MODELS
    # ----------------------------------------
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=200,
            random_state=42,
            n_jobs=-1
        ),
        "XGBoost": XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            tree_method="hist"     # faster training
        )
    }

    results = {}

    print("\nTraining Models...")

    # ----------------------------------------
    # TRAIN + EVALUATE
    # ----------------------------------------
    for name, model in models.items():
        print(f"\n  Training: {name}")
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        mae, rmse, r2 = evaluate_model(name, y_test, preds)
        results[name] = {"MAE": mae, "RMSE": rmse, "R2": r2}

    # ----------------------------------------
    # SAVE RESULTS CSV
    # ----------------------------------------
    results_df = pd.DataFrame(results).T
    results_df.to_csv("outputs/model_results.csv")
    print("\nResults saved → outputs/model_results.csv")

    # ----------------------------------------
    # SELECT BEST MODEL (by R²)
    # ----------------------------------------
    best_model_name = max(results, key=lambda x: results[x]["R2"])
    best_model = models[best_model_name]
    print(f"\nBest Model: {best_model_name}  (R² = {results[best_model_name]['R2']:.4f})")

    # ----------------------------------------
    # SAVE MODEL + FEATURE COLUMNS
    # ----------------------------------------
    joblib.dump(best_model, "models/model.pkl")
    joblib.dump(X.columns.tolist(), "models/feature_columns.pkl")
    print("Model saved → models/model.pkl")
    print("Feature columns saved → models/feature_columns.pkl")

    # ----------------------------------------
    # FEATURE IMPORTANCE (TREE MODELS ONLY)
    # ----------------------------------------
    if hasattr(best_model, "feature_importances_"):
        print("Generating Feature Importance chart...")
        importance = best_model.feature_importances_
        feat_series = pd.Series(importance, index=X.columns).sort_values(ascending=False).head(20)

        plt.figure(figsize=(10, 6))
        feat_series[::-1].plot(kind="barh", color="steelblue")
        plt.title(f"Top 20 Features — {best_model_name}")
        plt.xlabel("Importance Score")
        plt.tight_layout()
        plt.savefig("images/feature_importance.png", dpi=150)
        plt.close()
        print("Saved → images/feature_importance.png")

    # ----------------------------------------
    # ACTUAL VS PREDICTED PLOT
    # ----------------------------------------
    print("Generating Actual vs Predicted plot...")
    final_preds = best_model.predict(X_test)

    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, final_preds, alpha=0.5, color="steelblue")
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", lw=2)
    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    plt.title("Actual vs Predicted")
    plt.tight_layout()
    plt.savefig("images/actual_vs_predicted.png", dpi=150)
    plt.close()
    print("Saved → images/actual_vs_predicted.png")

    print("\nTRAINING COMPLETED SUCCESSFULLY!")
    return best_model