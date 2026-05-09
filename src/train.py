"""
train.py — Match Result Model Training (XGBoost)
==================================================
Fixed for: XGBoost 3.x + Pandas 3.x compatibility

Bugs fixed:
  1. Column names with special chars ([, ], <, >) crash XGBoost — cleaned BEFORE selection
  2. XGBoost 3.x + Pandas 3.x internal .dtype incompatibility — pass numpy array + feature_names explicitly
  3. MODEL_PARAMS.XGBOOST_DEFAULT had 'use_label_encoder' which is removed in XGBoost 2+ 

Run: python src/train.py
"""

import sys
import re
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
import joblib
import xgboost as xgb
from loguru import logger

from config import PATHS, MODEL_PARAMS

# ── Columns that must never be used as features ────────────────────────────────
EXCLUDE_COLS = {
    # Target variables
    'result', 'btts', 'over_25', 'over_corners',
    'total_goals', 'total_corners', 'goal_diff', 'result_numeric',
    # Post-match (leakage)
    'home_goals', 'away_goals', 'ht_home_goals', 'ht_away_goals',
    'home_elo_after', 'away_elo_after',
    # Metadata / identifiers
    'date', 'home_team', 'away_team', 'league_key', 'season',
    'season_label', 'league_name', 'country', 'referee',
    'ht_result', 'HTR', 'Time', 'league', 'Div',
    # Raw match stats (use rolling averages instead)
    'home_shots', 'away_shots', 'home_shots_ot', 'away_shots_ot',
    'home_corners', 'away_corners',
    'home_yellows', 'away_yellows', 'home_reds', 'away_reds',
    # Implied probabilities (use fair_prob_* instead)
    'implied_prob_home', 'implied_prob_draw', 'implied_prob_away', 'overround',
}


def sanitise_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace any character that is not a-z, A-Z, 0-9, or underscore with underscore.

    XGBoost 2.x+ rejects feature names containing [  ]  <  >  spaces or dots.
    This MUST be applied to the full DataFrame BEFORE selecting feature columns,
    so that the exclusion list still matches after cleaning.

    Returns a copy with renamed columns.
    """
    new_cols = {}
    seen = {}
    for col in df.columns:
        clean = re.sub(r'[^a-zA-Z0-9_]', '_', str(col))
        # Handle duplicates after cleaning (e.g. 'B365>2.5' and 'B365<2.5' both → 'B365_2_5')
        if clean in seen:
            seen[clean] += 1
            clean = f"{clean}_{seen[clean]}"
        else:
            seen[clean] = 0
        new_cols[col] = clean
    return df.rename(columns=new_cols)


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """
    Select numeric feature columns, excluding targets and metadata.
    Column names are already sanitised at this point.
    """
    # Sanitised versions of exclude set
    exclude_clean = {re.sub(r'[^a-zA-Z0-9_]', '_', c) for c in EXCLUDE_COLS}

    feature_cols = []
    for col in df.columns:
        if col in exclude_clean:
            continue
        # Only include numeric columns — skip any object/string columns
        if pd.api.types.is_numeric_dtype(df[col]):
            feature_cols.append(col)

    return feature_cols


def prepare_X_y(
    df: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    """
    Extract feature matrix and target vector.

    WHY CLEAN DATAFRAME (not numpy):
        XGBoost 3.x works correctly with Pandas DataFrames as long as
        column names contain only alphanumerics and underscores.
        The original crashes were caused by:
          1. Special chars in column names ([, ], <, >) — fixed by sanitise_column_names()
          2. Object/mixed-type columns — fixed by get_feature_columns() (numeric only)
          3. 'use_label_encoder' deprecated param — fixed in get_xgboost_params()
        Passing a clean DataFrame preserves feature_names_in_ on the model,
        which we need at prediction time.

    Returns:
        X            — clean float32 DataFrame, shape (n_samples, n_features)
        y            — int numpy array, shape (n_samples,)
        feature_cols — list of clean feature names (for model saving)
    """
    X = df[feature_cols].copy()

    # Force every column to numeric — any remaining strings become NaN
    X = X.apply(pd.to_numeric, errors='coerce')

    # Fill NaN with -999 sentinel (XGBoost handles missing values natively,
    # but -999 makes missing data explicit so tree splits can isolate it)
    X = X.fillna(-999)

    # Cast to float32 — smaller memory footprint
    X = X.astype(np.float32)

    # Target: H=0, D=1, A=2
    y = df['result'].map({'H': 0, 'D': 1, 'A': 2}).to_numpy(dtype=np.int32)

    return X, y, feature_cols


def get_xgboost_params() -> dict:
    """
    Build XGBoost parameters, stripping keys removed in newer versions.

    'use_label_encoder' was removed in XGBoost 2.0 and crashes if included.
    We build params defensively so config.py doesn't need to change.
    """
    params = dict(MODEL_PARAMS.XGBOOST_DEFAULT)

    # Remove deprecated / removed parameters
    for removed_key in ['use_label_encoder']:
        params.pop(removed_key, None)

    # Force correct objective for multi-class
    params['objective']   = 'multi:softprob'
    params['eval_metric'] = 'mlogloss'
    params['num_class']   = 3

    return params


def train_model(
    seasons: list[str] | None = None,
    save: bool = True,
) -> xgb.XGBClassifier:
    """
    Full training pipeline for the Match Result (1X2) model.

    Args:
        seasons: List of season codes to train on (default: all training seasons)
        save:    Whether to save the model to disk

    Returns:
        Trained XGBClassifier
    """
    logger.info("Loading features parquet ...")
    df = pd.read_parquet(PATHS.PROCESSED / "features_master.parquet")
    logger.info(f"Loaded {len(df):,} matches, {df.shape[1]} raw columns")

    # ── Step 1: Season filter ──────────────────────────────────────────────
    if seasons:
        df = df[df['season'].isin(seasons)].copy()
        logger.info(f"Filtered to seasons {seasons}: {len(df):,} matches")

    # ── Step 2: Sanitise ALL column names before anything else ─────────────
    # Must happen first so EXCLUDE_COLS matching works correctly
    df = sanitise_column_names(df)

    # ── Step 3: Drop rows with no result (shouldn't happen, but be safe) ───
    df = df[df['result'].isin(['H', 'D', 'A'])].copy()
    logger.info(f"Valid result rows: {len(df):,}")

    # ── Step 4: Select feature columns ────────────────────────────────────
    feature_cols = get_feature_columns(df)
    logger.info(f"Training with {len(feature_cols)} numeric features")

    if len(feature_cols) == 0:
        raise ValueError("No feature columns found after filtering. Check EXCLUDE_COLS and parquet schema.")

    # ── Step 5: Build X (numpy) and y ────────────────────────────────────
    X, y, feature_cols = prepare_X_y(df, feature_cols)
    logger.info(f"X shape: {X.shape} | y distribution: {dict(zip(*np.unique(y, return_counts=True)))}")

    # ── Step 6: Train XGBoost ─────────────────────────────────────────────
    params = get_xgboost_params()
    logger.info(f"XGBoost params: n_estimators={params['n_estimators']}, "
                f"max_depth={params['max_depth']}, lr={params['learning_rate']}")

    model = xgb.XGBClassifier(**params)

    # Pass clean DataFrame — XGBoost 3.x reads feature names from column headers.
    # Column names were sanitised so no special chars remain.
    model.fit(X, y, verbose=False)

    logger.success(f"Model trained successfully on {len(y):,} samples")

    # ── Step 7: Quick sanity check ────────────────────────────────────────
    sample_pred = model.predict_proba(X[:5])
    logger.info(f"Sample probabilities (first 5 rows):\n{np.round(sample_pred, 3)}")
    assert sample_pred.shape[1] == 3, "Expected 3 output classes (H/D/A)"
    assert np.allclose(sample_pred.sum(axis=1), 1.0, atol=1e-5), "Probabilities must sum to 1"
    logger.success("Sanity checks passed.")

    # ── Step 8: Save model and feature list ──────────────────────────────
    if save:
        PATHS.MODELS.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, PATHS.MODEL_MATCH_RESULT)
        # Save feature names alongside model — needed at prediction time
        feature_path = PATHS.MODELS / "match_result_features.pkl"
        joblib.dump(feature_cols, feature_path)
        logger.success(f"Model saved: {PATHS.MODEL_MATCH_RESULT}")
        logger.success(f"Feature list saved: {feature_path}")

    return model


if __name__ == "__main__":
    import argparse
    PATHS.create_all()

    parser = argparse.ArgumentParser(description="Train the Match Result XGBoost model")
    parser.add_argument("--seasons", nargs="+", default=None,
                        help="Seasons to train on e.g. --seasons 2223 2324")
    parser.add_argument("--no-save", action="store_true",
                        help="Train without saving model to disk")
    args = parser.parse_args()

    model = train_model(seasons=args.seasons, save=not args.no_save)
    logger.success("train.py complete.")