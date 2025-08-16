import sys
import os
import argparse
import sqlite3
import json
from typing import Tuple, List

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV, LassoCV
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import SIGNAL_WEIGHT_PROFILES, FEATURE_TOGGLES, DB_PATH, WEIGHTS_OUTPUT_PATH
from config.labels import FINAL_COLUMN_ORDER

# === Load Data ===
def load_backtest_data(db_path: str, return_target: str) -> pd.DataFrame:
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql("SELECT * FROM signals", conn)
    # If the requested target isn't present yet (e.g., too recent), return empty
    if return_target not in df.columns:
        return pd.DataFrame()
    df = df.dropna(subset=[return_target])
    return df

# === Prepare Features ===
def extract_features_targets(df: pd.DataFrame, return_target: str) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
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

# === Train Model (Ridge or Lasso) with scaling and imputation ===
def build_model(model_type: str = "ridge") -> Pipeline:
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[("num", numeric_transformer, slice(0, None))]
    )

    if model_type == "lasso":
        estimator = LassoCV(alphas=np.logspace(-3, 2, 40), cv=5, random_state=42, max_iter=5000)
    else:
        estimator = RidgeCV(alphas=np.logspace(-3, 3, 50), cv=5, scoring="r2")

    return Pipeline(steps=[("preprocessor", preprocessor), ("model", estimator)])

# === Save New Profile ===
def save_weights(weights: np.ndarray, feature_cols: List[str], output_path: str) -> None:
    profile = {col: round(float(weights[i]), 4) for i, col in enumerate(feature_cols)}
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(profile, f, indent=2)
    print(f"Saved ML-optimized weights to {output_path}")

# === Main ===
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default="3D Return", help="Target return column (e.g., '3D Return' or '10D Return')")
    parser.add_argument("--model", choices=["ridge", "lasso"], default="ridge", help="Model type for weight learning")
    parser.add_argument("--test-size", type=float, default=0.2, help="Holdout fraction for validation")
    args = parser.parse_args()
    return_target = args.target

    print(f"=== SCORING_TRAINER STARTED: Target = {return_target} ===")

    df = load_backtest_data(DB_PATH, return_target)
    if df.empty:
        print(f"No '{return_target}' data available yet. Skipping training.")
        return
    X, y, feature_cols = extract_features_targets(df, return_target)

    if X.empty or y.empty:
        print("No data available for training. Check signal availability or filters.")
        return

    print(f"Training on {X.shape[0]} samples and {X.shape[1]} features.")

    # Split for honest validation
    X_train, X_test, y_train, y_test = train_test_split(X.values, y.values, test_size=args.test_size, random_state=42)
    pipe = build_model(args.model)
    pipe.fit(X_train, y_train)

    # Evaluate
    pred_train = pipe.predict(X_train)
    pred_test = pipe.predict(X_test)
    print(f"Train R²: {r2_score(y_train, pred_train):.4f} | MAE: {mean_absolute_error(y_train, pred_train):.4f}")
    print(f"Test  R²: {r2_score(y_test, pred_test):.4f} | MAE: {mean_absolute_error(y_test, pred_test):.4f}")

    # Extract weights back from linear model (after scaling). We approximate by fitting a plain linear model on transformed data
    # so that coef_ maps to original feature ordering after the preprocessor.
    # Since our pipeline is numeric-only and standardized, we can grab the model coef and map back.
    model = pipe.named_steps["model"]
    if hasattr(model, "coef_"):
        coefs = model.coef_
        save_weights(coefs, feature_cols, WEIGHTS_OUTPUT_PATH)
    else:
        print("Model does not expose coefficients; skipping weight export.")
    print("=== SCORING_TRAINER COMPLETED SUCCESSFULLY ===")

if __name__ == "__main__":
    main()
