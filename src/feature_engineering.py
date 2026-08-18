"""
feature_engineering.py — Rolling Feature Builder
==================================================
Transforms raw match data into model-ready features.

Works with the project's actual schema:
    - home_goals, away_goals (numeric)
    - result (H / D / A)
    - league_key, season, date
    - home_team, away_team

Generates:
    - Rolling form averages (goals, shots, corners, cards)
    - Home/away split features
    - Goal difference trends
    - BTTS / Over 2.5 rolling rates
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import PATHS, FEATURES, LOGGING

logger.remove()
logger.add(sys.stderr, level=LOGGING.LEVEL, format=LOGGING.FORMAT, colorize=True)


def _ensure_columns(df: pd.DataFrame, required: list[str]) -> pd.DataFrame:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return df


def build_features(df: pd.DataFrame, windows: Optional[list[int]] = None) -> pd.DataFrame:
    """
    Build rolling features for all teams in the DataFrame.

    Args:
        df:      Master DataFrame with home_goals, away_goals, result, etc.
        windows: Rolling window sizes (default from config: [3, 5, 10])

    Returns:
        DataFrame with added rolling feature columns.
    """
    if windows is None:
        windows = FEATURES.ROLLING_WINDOWS

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    required = ["date", "home_team", "away_team", "home_goals", "away_goals", "result"]
    _ensure_columns(df, required)

    df.sort_values(["league_key", "date"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    logger.info(f"Building rolling features (windows={windows}) for {len(df):,} matches")

    for w in windows:
        suffix = f"roll{w}"

        df[f"home_goals_{suffix}"] = (
            df.groupby("home_team")["home_goals"]
            .transform(lambda x: x.shift().rolling(window=w, min_periods=1).mean())
        )
        df[f"away_goals_{suffix}"] = (
            df.groupby("away_team")["away_goals"]
            .transform(lambda x: x.shift().rolling(window=w, min_periods=1).mean())
        )

        df[f"home_conceded_{suffix}"] = (
            df.groupby("home_team")["away_goals"]
            .transform(lambda x: x.shift().rolling(window=w, min_periods=1).mean())
        )
        df[f"away_conceded_{suffix}"] = (
            df.groupby("away_team")["home_goals"]
            .transform(lambda x: x.shift().rolling(window=w, min_periods=1).mean())
        )

        if "btts" in df.columns:
            df[f"home_btts_rate_{suffix}"] = (
                df.groupby("home_team")["btts"]
                .transform(lambda x: x.shift().rolling(window=w, min_periods=1).mean())
            )
            df[f"away_btts_rate_{suffix}"] = (
                df.groupby("away_team")["btts"]
                .transform(lambda x: x.shift().rolling(window=w, min_periods=1).mean())
            )

        if "over_25" in df.columns:
            df[f"home_over25_rate_{suffix}"] = (
                df.groupby("home_team")["over_25"]
                .transform(lambda x: x.shift().rolling(window=w, min_periods=1).mean())
            )
            df[f"away_over25_rate_{suffix}"] = (
                df.groupby("away_team")["over_25"]
                .transform(lambda x: x.shift().rolling(window=w, min_periods=1).mean())
            )

        if "home_corners" in df.columns:
            df[f"home_corners_{suffix}"] = (
                df.groupby("home_team")["home_corners"]
                .transform(lambda x: x.shift().rolling(window=w, min_periods=1).mean())
            )
            df[f"away_corners_{suffix}"] = (
                df.groupby("away_team")["away_corners"]
                .transform(lambda x: x.shift().rolling(window=w, min_periods=1).mean())
            )

        if "home_yellows" in df.columns:
            df[f"home_cards_{suffix}"] = (
                df.groupby("home_team")["home_yellows"]
                .transform(lambda x: x.shift().rolling(window=w, min_periods=1).mean())
            )
            df[f"away_cards_{suffix}"] = (
                df.groupby("away_team")["away_yellows"]
                .transform(lambda x: x.shift().rolling(window=w, min_periods=1).mean())
            )

    result_map = {"H": 1, "D": 0, "A": -1}
    df["result_value"] = df["result"].map(result_map).fillna(0)
    for w in [3, 5]:
        df[f"form_index_{w}"] = (
            df.groupby("home_team")["result_value"]
            .transform(lambda x: x.shift().rolling(window=w, min_periods=1).sum())
        )
        df[f"away_form_index_{w}"] = (
            df.groupby("away_team")["result_value"]
            .transform(lambda x: x.shift().rolling(window=w, min_periods=1).sum())
        )

    df = df.fillna(0)

    logger.success(f"Feature engineering complete: {df.shape[1]} columns")
    return df


def get_feature_list(df: pd.DataFrame, exclude: Optional[set[str]] = None) -> list[str]:
    """
    Return numeric feature columns suitable for model training.

    Excludes targets, metadata, and raw post-match stats.
    """
    if exclude is None:
        exclude = {
            "result", "btts", "over_25", "over_corners",
            "total_goals", "total_corners", "goal_diff", "result_numeric",
            "home_goals", "away_goals", "ht_home_goals", "ht_away_goals",
            "home_elo_after", "away_elo_after",
            "date", "home_team", "away_team", "league_key", "season",
            "season_label", "league_name", "country", "referee",
            "ht_result", "HTR", "Time", "league", "Div",
            "home_shots", "away_shots", "home_shots_ot", "away_shots_ot",
            "home_corners", "away_corners",
            "home_yellows", "away_yellows", "home_reds", "away_reds",
            "implied_prob_home", "implied_prob_draw", "implied_prob_away",
            "overround", "result_value",
        }

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    return [c for c in numeric_cols if c not in exclude]


if __name__ == "__main__":
    df = pd.read_parquet(PATHS.PROCESSED / "master_with_elo.parquet")
    features_df = build_features(df)
    features_df.to_parquet(PATHS.PROCESSED / "features_master.parquet", index=False)
    logger.success(f"Saved features_master.parquet ({len(features_df):,} rows)")
