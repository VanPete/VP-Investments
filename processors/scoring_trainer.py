import sys
import os
import argparse
import sqlite3
import pandas as pd
import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score, mean_absolute_error
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import SIGNAL_WEIGHT_PROFILES, FEATURE_TOGGLES, DB_PATH, WEIGHTS_OUTPUT_PATH
from config.labels import FINAL_COLUMN_ORDER

# === Load Data ===
def load_backtest_data(db_path, return_target):
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql("SELECT * FROM signals", conn)
    df = df.dropna(subset=[return_target])
    return df

# === Prepare Features ===
def extract_features_targets(df, return_target):
    excluded = {"Ticker", "Run Datetime", return_target}
    active_features = [col for col in FEATURE_TOGGLES if FEATURE_TOGGLES[col]]
    missing_features = set(active_features) - set(df.columns)

    if missing_features:
        print(f"Warning: Missing features not found in data: {missing_features}")

    feature_cols = [col for col in active_features if col in df.columns and col not in excluded]
    df = df[feature_cols + [return_target]].copy()
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.dropna()
    X = df[feature_cols].copy()
    y = df[return_target].copy()
    return X, y, feature_cols

# === Train Ridge Regression ===
def train_model(X, y):
    model = RidgeCV(alphas=np.logspace(-3, 3, 50), cv=5, scoring="r2")
    model.fit(X, y)
    return model

# === Save New Profile ===
def save_weights(weights, feature_cols, output_path):
    profile = {col: round(weights[i], 4) for i, col in enumerate(feature_cols)}
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(profile, f, indent=2)
    print(f"Saved ML-optimized weights to {output_path}")

# === Main ===
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default="3D Return", help="Target return column (e.g., '3D Return' or '10D Return')")
    args = parser.parse_args()
    return_target = args.target

    print(f"=== SCORING_TRAINER STARTED: Target = {return_target} ===")

    df = load_backtest_data(DB_PATH, return_target)
    X, y, feature_cols = extract_features_targets(df, return_target)

    if X.empty or y.empty:
        print("No data available for training. Check signal availability or filters.")
        return

    print(f"Training on {X.shape[0]} samples and {X.shape[1]} features.")

    model = train_model(X, y)
    preds = model.predict(X)

    r2 = r2_score(y, preds)
    mae = mean_absolute_error(y, preds)

    print(f"R² Score: {r2:.4f}")
    print(f"Mean Absolute Error: {mae:.4f}")

    save_weights(model.coef_, feature_cols, WEIGHTS_OUTPUT_PATH)
    print("=== SCORING_TRAINER COMPLETED SUCCESSFULLY ===")

if __name__ == "__main__":
    main()
