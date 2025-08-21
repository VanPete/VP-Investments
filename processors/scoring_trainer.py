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
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import SIGNAL_WEIGHT_PROFILES, FEATURE_TOGGLES, DB_PATH, WEIGHTS_OUTPUT_PATH
from utils.db import insert_experiment, insert_metric
from datetime import datetime, timezone
import uuid
from config.labels import FINAL_COLUMN_ORDER

# === Load Data ===
def load_backtest_data(db_path: str, return_target: str) -> pd.DataFrame:
    """Load training data from normalized tables when available, else fall back to wide signals."""
    with sqlite3.connect(db_path) as conn:
        try:
            # Pivot features wide per (run_id, ticker)
            f = pd.read_sql("SELECT run_id, ticker, key, value FROM features", conn)
            if not f.empty:
                f_piv = f.pivot_table(index=["run_id", "ticker"], columns="key", values="value", aggfunc="first")
                f_piv.reset_index(inplace=True)
                # Labels for the requested window
                win = return_target.replace(" Return", "")  # e.g., "3D"
                l = pd.read_sql("SELECT run_id, ticker, window, fwd_return FROM labels", conn)
                l = l[l["window"].astype(str) == win]
                l = l.rename(columns={"fwd_return": return_target})
                df = f_piv.merge(l[["run_id", "ticker", return_target]], on=["run_id", "ticker"], how="inner")
                df = df.dropna(subset=[return_target])
                if not df.empty:
                    # Add ticker and run datetime if available from signals for convenience
                    try:
                        sig = pd.read_sql("SELECT \"Run ID\" as run_id, \"Ticker\" as ticker, \"Run Datetime\" FROM signals", conn)
                        sig["Run Datetime"] = pd.to_datetime(sig["Run Datetime"], errors='coerce')
                        df = df.merge(sig, on=["run_id", "ticker"], how="left")
                    except Exception:
                        pass
                    return df
        except Exception:
            pass
        # Fallback: use wide signals table
        df = pd.read_sql("SELECT * FROM signals", conn)
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
def _rolling_cv_splits(n_splits: int, embargo: int, n_samples: int):
    """Generate rolling time-based splits with an embargo window on the training tail.

    embargo is the number of samples at the end of train to drop (to reduce leakage).
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    for tr, te in tscv.split(range(n_samples)):
        if embargo > 0:
            tr = tr[:-embargo] if len(tr) > embargo else tr[:0]
        yield tr, te


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default="3D Return", help="Target return column (e.g., '3D Return' or '10D Return')")
    parser.add_argument("--model", choices=["ridge", "lasso"], default="ridge", help="Model type for weight learning")
    parser.add_argument("--test-size", type=float, default=0.2, help="Holdout fraction for validation (ignored if --rolling-cv)")
    parser.add_argument("--rolling-cv", action="store_true", help="Use rolling time-split CV with embargo instead of random holdout")
    parser.add_argument("--cv-splits", type=int, default=5, help="Number of rolling splits when --rolling-cv is set")
    parser.add_argument("--embargo", type=int, default=5, help="Samples to drop from the end of train to reduce leakage in rolling CV")
    args = parser.parse_args()
    return_target = args.target

    print(f"=== SCORING_TRAINER STARTED: Target = {return_target} ===")
    exp_id = f"exp-{uuid.uuid4()}"
    started = datetime.now(timezone.utc).isoformat()
    try:
        insert_experiment(
            exp_id=exp_id,
            run_id=None,
            profile="trainer",
            params_json=json.dumps({"target": return_target, "model": args.model, "test_size": args.test_size}),
            code_version=os.getenv("CODE_VERSION", "local"),
            started_at=started,
            ended_at=None,
            notes=None,
        )
    except Exception:
        pass

    df = load_backtest_data(DB_PATH, return_target)
    if df.empty:
        print(f"No '{return_target}' data available yet. Skipping training.")
        return
    X, y, feature_cols = extract_features_targets(df, return_target)

    if X.empty or y.empty:
        print("No data available for training. Check signal availability or filters.")
        return

    print(f"Training on {X.shape[0]} samples and {X.shape[1]} features.")

    if args.rolling_cv:
        # Rolling CV with embargo
        print(f"Rolling CV: splits={args.cv_splits}, embargo={args.embargo}")
        pipe = build_model(args.model)
        fold_metrics = []
        for i, (tr, te) in enumerate(_rolling_cv_splits(args.cv_splits, args.embargo, len(y))):
            if len(tr) == 0 or len(te) == 0:
                continue
            pipe.fit(X.values[tr], y.values[tr])
            pred_tr = pipe.predict(X.values[tr])
            pred_te = pipe.predict(X.values[te])
            m = {
                "fold": i,
                "train_r2": float(r2_score(y.values[tr], pred_tr)),
                "train_mae": float(mean_absolute_error(y.values[tr], pred_tr)),
                "test_r2": float(r2_score(y.values[te], pred_te)),
                "test_mae": float(mean_absolute_error(y.values[te], pred_te)),
            }
            fold_metrics.append(m)
            print(f"Fold {i}: Train R² {m['train_r2']:.4f} | Test R² {m['test_r2']:.4f} | Test MAE {m['test_mae']:.4f}")
            try:
                ts = datetime.now(timezone.utc).isoformat()
                insert_metric(exp_id, f"fold_{i}_train_r2", m["train_r2"], None, ts)
                insert_metric(exp_id, f"fold_{i}_train_mae", m["train_mae"], None, ts)
                insert_metric(exp_id, f"fold_{i}_test_r2", m["test_r2"], None, ts)
                insert_metric(exp_id, f"fold_{i}_test_mae", m["test_mae"], None, ts)
            except Exception:
                pass
        # Use last trained model for weights export
        model = pipe.named_steps["model"]
        if hasattr(model, "coef_"):
            save_weights(model.coef_, feature_cols, WEIGHTS_OUTPUT_PATH)
        else:
            print("Model does not expose coefficients; skipping weight export.")
    else:
        # Random holdout split (default)
        X_train, X_test, y_train, y_test = train_test_split(X.values, y.values, test_size=args.test_size, random_state=42)
        pipe = build_model(args.model)
        pipe.fit(X_train, y_train)

        # Evaluate
        pred_train = pipe.predict(X_train)
        pred_test = pipe.predict(X_test)
        tr_r2 = r2_score(y_train, pred_train)
        tr_mae = mean_absolute_error(y_train, pred_train)
        te_r2 = r2_score(y_test, pred_test)
        te_mae = mean_absolute_error(y_test, pred_test)
        print(f"Train R²: {tr_r2:.4f} | MAE: {tr_mae:.4f}")
        print(f"Test  R²: {te_r2:.4f} | MAE: {te_mae:.4f}")
        # Log metrics to DB
        try:
            ts = datetime.now(timezone.utc).isoformat()
            insert_metric(exp_id, "train_r2", float(tr_r2), None, ts)
            insert_metric(exp_id, "train_mae", float(tr_mae), None, ts)
            insert_metric(exp_id, "test_r2", float(te_r2), None, ts)
            insert_metric(exp_id, "test_mae", float(te_mae), None, ts)
        except Exception:
            pass

        # Export weights if available
        model = pipe.named_steps["model"]
        if hasattr(model, "coef_"):
            save_weights(model.coef_, feature_cols, WEIGHTS_OUTPUT_PATH)
        else:
            print("Model does not expose coefficients; skipping weight export.")
    # Mark experiment end
    try:
        insert_experiment(
            exp_id=exp_id,
            run_id=None,
            profile="trainer",
            params_json=None,
            code_version=None,
            started_at=None,
            ended_at=datetime.now(timezone.utc).isoformat(),
            notes=None,
        )
    except Exception:
        pass
    print("=== SCORING_TRAINER COMPLETED SUCCESSFULLY ===")

if __name__ == "__main__":
    main()
