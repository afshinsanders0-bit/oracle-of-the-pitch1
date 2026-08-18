"""
trainCurrentOnly.py — Train Models Using Only Current Data & H2H
==================================================================
Trains models using ONLY:
  1. Current match statistics (shots, corners, possession)
  2. Head-to-head history between teams

NO historical form data, rolling averages, or team strength metrics.

USAGE:
    python src/trainCurrentOnly.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import joblib
from loguru import logger
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score

import xgboost as xgb
import lightgbm as lgb

from config import PATHS, SEASONS, MODEL_PARAMS, RANDOM_SEED
from src.predictCurrentOnly import get_h2h_features

logger.remove()
logger.add(sys.stderr, level="INFO", format="{message}")

np.random.seed(RANDOM_SEED)


# ─────────────────────────────────────────────
# Column auto-detection
# ─────────────────────────────────────────────

def resolve_columns(df: pd.DataFrame) -> dict:
    """
    Auto-detect actual column names in df for each canonical feature name.
    Returns a dict mapping canonical -> actual column name (only found keys included).
    """
    candidates = {
        "home_shots":      ["home_shots", "HS", "HomeShots", "home_shots_total"],
        "away_shots":      ["away_shots", "AS", "AwayShots", "away_shots_total"],
        "home_corners":    ["home_corners", "HC", "HomeCorners", "home_corners_total"],
        "away_corners":    ["away_corners", "AC", "AwayCorners", "away_corners_total"],
        "home_possession": ["home_possession", "HBP", "home_poss", "HomePossession", "home_possession_pct"],
        "away_possession": ["away_possession", "ABP", "away_poss", "AwayPossession", "away_possession_pct"],
    }
    col_map = {}
    for canonical, options in candidates.items():
        for opt in options:
            if opt in df.columns:
                col_map[canonical] = opt
                break
    return col_map


# ─────────────────────────────────────────────
# Build training data
# ─────────────────────────────────────────────

def build_training_data_current_only(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list]:
    """
    Build training data using ONLY current match stats and H2H.

    Features included:
      - home_shots, away_shots, shot_diff
      - home_corners, away_corners, corner_diff
      - home_possession, away_possession, possession_diff  (if available)
      - H2H features

    Returns:
        (X, y, feature_names)
    """
    logger.info("Building training data (current only + H2H)...")

    col_map = resolve_columns(df)
    logger.info(f"Detected column mapping: {col_map}")

    # Shots and corners are hard-required
    hard_required = ["home_shots", "away_shots", "home_corners", "away_corners"]
    missing = [k for k in hard_required if k not in col_map]
    if missing:
        raise KeyError(
            f"Could not find columns for: {missing}.\n"
            f"Available DataFrame columns: {df.columns.tolist()}"
        )

    actual_required = [col_map[k] for k in hard_required]
    df = df.dropna(subset=actual_required).copy()
    logger.info(f"Matches with current stats: {len(df):,}")

    # H2H features
    h2h_features_list = []
    for idx, row in df.iterrows():
        h2h = get_h2h_features(
            row["home_team"],
            row["away_team"],
            df[df["date"] < row["date"]],
            league_key=row.get("league_key")
        )
        h2h_features_list.append(h2h)

    h2h_df = pd.DataFrame(h2h_features_list, index=df.index)

    # Base features (shots + corners)
    feat = {
        "home_shots":   df[col_map["home_shots"]].values,
        "away_shots":   df[col_map["away_shots"]].values,
        "home_corners": df[col_map["home_corners"]].values,
        "away_corners": df[col_map["away_corners"]].values,
        "shot_diff":    (df[col_map["home_shots"]] - df[col_map["away_shots"]]).values,
        "corner_diff":  (df[col_map["home_corners"]] - df[col_map["away_corners"]]).values,
    }

    # Possession — optional
    has_possession = "home_possession" in col_map and "away_possession" in col_map
    if has_possession:
        feat["home_possession"] = df[col_map["home_possession"]].values
        feat["away_possession"] = df[col_map["away_possession"]].values
        feat["possession_diff"] = (
            df[col_map["home_possession"]] - df[col_map["away_possession"]]
        ).values
        logger.info("Possession features included.")
    else:
        logger.warning("Possession columns not found — skipping possession features.")

    X = pd.DataFrame(feat, index=df.index)

    for col in h2h_df.columns:
        X[col] = h2h_df[col].values

    y = df["result"].map({"H": 0, "D": 1, "A": 2})
    feature_names = X.columns.tolist()

    logger.info(f"Training shape: X={X.shape}, y={y.shape}")
    logger.info(f"Features: {feature_names}")

    return X, y, feature_names


def _cv_score(model_cls, model_kwargs, X, y, n_splits=5) -> tuple[float, float]:
    """Run TimeSeriesSplit CV and return (mean_acc, std_acc)."""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    scores = []
    for train_idx, test_idx in tscv.split(X):
        m = model_cls(**model_kwargs)
        m.fit(X.iloc[train_idx], y[train_idx] if isinstance(y, np.ndarray) else y.iloc[train_idx])
        scores.append(accuracy_score(
            y[test_idx] if isinstance(y, np.ndarray) else y.iloc[test_idx],
            m.predict(X.iloc[test_idx])
        ))
    return float(np.mean(scores)), float(np.std(scores))


# ─────────────────────────────────────────────
# Model trainers
# ─────────────────────────────────────────────

def train_match_result(X: pd.DataFrame, y: pd.Series, feature_names: list):
    """Train match result model (1X2)."""
    logger.info("Training Match Result model (XGBoost)...")

    kwargs = dict(
        n_estimators=300, max_depth=4, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        random_state=RANDOM_SEED, n_jobs=-1, eval_metric="mlogloss"
    )
    model = xgb.XGBClassifier(**kwargs)
    model.fit(X, y)

    mean_acc, std_acc = _cv_score(xgb.XGBClassifier, kwargs, X, y)
    logger.info(f"CV Accuracy: {mean_acc:.1%} (+/- {std_acc:.1%})")

    joblib.dump(model, PATHS.MODEL_MATCH_RESULT)
    joblib.dump(feature_names, PATHS.MODELS / "match_result_current_features.pkl")
    logger.success(f"Saved: {PATHS.MODEL_MATCH_RESULT.name}")
    return model


def train_btts(df: pd.DataFrame, X: pd.DataFrame, feature_names: list):
    """Train BTTS model (Both Teams to Score)."""
    if "btts" not in df.columns:
        logger.warning("BTTS column not found - skipping")
        return None

    logger.info("Training BTTS model (LightGBM)...")
    y_btts = df.loc[X.index, "btts"].values

    kwargs = dict(
        n_estimators=300, max_depth=4, learning_rate=0.1,
        random_state=RANDOM_SEED, n_jobs=-1, verbose=-1
    )
    model = lgb.LGBMClassifier(**kwargs)
    model.fit(X, y_btts)

    mean_acc, std_acc = _cv_score(lgb.LGBMClassifier, kwargs, X, y_btts)
    logger.info(f"CV Accuracy: {mean_acc:.1%} (+/- {std_acc:.1%})")

    joblib.dump(model, PATHS.MODEL_BTTS)
    joblib.dump(feature_names, PATHS.MODELS / "btts_current_features.pkl")
    logger.success(f"Saved: {PATHS.MODEL_BTTS.name}")
    return model


def train_over_under(df: pd.DataFrame, X: pd.DataFrame, feature_names: list):
    """Train Over/Under 2.5 model."""
    if "over_25" not in df.columns:
        logger.warning("Over/Under column not found - skipping")
        return None

    logger.info("Training Over/Under 2.5 model (XGBoost)...")
    y_ou = df.loc[X.index, "over_25"].values

    kwargs = dict(
        n_estimators=300, max_depth=4, learning_rate=0.1,
        random_state=RANDOM_SEED, n_jobs=-1, eval_metric="logloss"
    )
    model = xgb.XGBClassifier(**kwargs)
    model.fit(X, y_ou)

    mean_acc, std_acc = _cv_score(xgb.XGBClassifier, kwargs, X, y_ou)
    logger.info(f"CV Accuracy: {mean_acc:.1%} (+/- {std_acc:.1%})")

    joblib.dump(model, PATHS.MODEL_OVER_UNDER)
    joblib.dump(feature_names, PATHS.MODELS / "over_under_current_features.pkl")
    logger.success(f"Saved: {PATHS.MODEL_OVER_UNDER.name}")
    return model


def train_corners(df: pd.DataFrame, X: pd.DataFrame, feature_names: list):
    """
    Train Corners Over 9.5 model.

    IMPORTANT: corner features (home_corners, away_corners, corner_diff) are
    EXCLUDED here because they directly sum to the target (total_corners > 9.5),
    causing data leakage and a falsely perfect CV score (~99.9%).
    """
    if "total_corners" not in df.columns:
        logger.warning("Corners column not found - skipping")
        return None

    logger.info("Training Corners model (LightGBM)...")

    # Drop corner-related columns to prevent leakage
    leaky_cols = ["home_corners", "away_corners", "corner_diff"]
    corners_features = [f for f in feature_names if f not in leaky_cols]
    X_corners = X[corners_features]

    logger.info(f"Corners model features (leaky cols removed): {corners_features}")

    y_corners = (df.loc[X.index, "total_corners"] > 9.5).astype(int).values

    kwargs = dict(
        n_estimators=300, max_depth=4, learning_rate=0.1,
        random_state=RANDOM_SEED, n_jobs=-1, verbose=-1
    )
    model = lgb.LGBMClassifier(**kwargs)
    model.fit(X_corners, y_corners)

    mean_acc, std_acc = _cv_score(lgb.LGBMClassifier, kwargs, X_corners, y_corners)
    logger.info(f"CV Accuracy: {mean_acc:.1%} (+/- {std_acc:.1%})")

    joblib.dump(model, PATHS.MODEL_CORNERS)
    joblib.dump(corners_features, PATHS.MODELS / "corners_current_features.pkl")
    logger.success(f"Saved: {PATHS.MODEL_CORNERS.name}")
    return model


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

def main():
    """Train all models using current-only approach."""
    logger.info("="*60)
    logger.info("Training Current-Only Models (No Historical Form)")
    logger.info("="*60)

    try:
        df = pd.read_parquet(PATHS.PROCESSED / "master.parquet")
    except FileNotFoundError:
        logger.error("Features not found. Run: python src/update_data.py")
        return

    df = df[df["season"].isin(SEASONS.TRAIN)].copy()
    df.sort_values("date", inplace=True)
    logger.info(f"Training data: {len(df):,} matches")

    X, y, feature_names = build_training_data_current_only(df)

    logger.info("\n" + "="*60)
    train_match_result(X, y, feature_names)

    logger.info("\n" + "="*60)
    train_btts(df, X, feature_names)

    logger.info("\n" + "="*60)
    train_over_under(df, X, feature_names)

    logger.info("\n" + "="*60)
    train_corners(df, X, feature_names)

    logger.info("\n" + "="*60)
    logger.success("All models trained!")
    logger.info("Run: python src/predictCurrentOnly.py")


if __name__ == "__main__":
    main()