"""
predictCurrentOnly.py — Match Result Prediction Using Only Current Data & H2H
==================================================================================
APPROACH:
    1. Current match data (shots, corners, possession, etc.) for THIS match only
    2. Head-to-Head history between the two teams
    3. No rolling averages, no historical form
    4. Lightweight, interpretable, fast

FEATURES:
    Current Match:
    - Home/Away team names
    - Home/Away shots
    - Home/Away corners
    - Home/Away possession (optional — skipped gracefully if not available)

    H2H History:
    - Win/loss/draw ratio
    - Goals scored/conceded ratio
    - Avg goals per match
    - Home advantage in H2H

MODELS:
    - Match Result (1X2): XGBoost
    - BTTS: LightGBM
    - Over/Under 2.5: XGBoost
    - Corners: LightGBM

USAGE:
    from src.predictCurrentOnly import predict_match_current_only

    result = predict_match_current_only(
        home_team="Arsenal",
        away_team="Chelsea",
        home_shots=15,
        away_shots=8,
        home_corners=6,
        away_corners=4,
        home_possession=58.5,   # optional — pass None if unavailable
        away_possession=41.5,   # optional — pass None if unavailable
        league_key="EPL",
        features_df=df          # Historical data for H2H only
    )
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import joblib
from loguru import logger
from typing import Dict, Any, Optional

from config import PATHS, LEAGUES, FEATURES, KELLY

logger.remove()
logger.add(sys.stderr, level="INFO", format="{message}")


# ═══════════════════════════════════════════════════════
# SECTION 1 — H2H FEATURE EXTRACTION
# ═══════════════════════════════════════════════════════

def get_h2h_features(
    home_team: str,
    away_team: str,
    features_df: pd.DataFrame,
    league_key: Optional[str] = None
) -> Dict[str, float]:
    """
    Extract head-to-head features between two teams.

    Features calculated from ALL historical matches between these teams
    (regardless of league or season).

    Args:
        home_team: Home team name
        away_team: Away team name
        features_df: Historical DataFrame (any league, any season)
        league_key: Optional - filter to specific league

    Returns:
        Dict with H2H features like h2h_home_wins, h2h_avg_goals, etc.
    """
    if league_key:
        df = features_df[features_df["league_key"] == league_key].copy()
    else:
        df = features_df.copy()

    h2h = df[
        ((df["home_team"] == home_team) & (df["away_team"] == away_team)) |
        ((df["home_team"] == away_team) & (df["away_team"] == home_team))
    ].copy()

    if h2h.empty:
        return {
            "h2h_matches": 0,
            "h2h_home_wins": 0,
            "h2h_away_wins": 0,
            "h2h_draws": 0,
            "h2h_home_goals_for": 1.5,
            "h2h_home_goals_against": 1.0,
            "h2h_away_goals_for": 1.0,
            "h2h_away_goals_against": 1.5,
            "h2h_avg_total_goals": 2.5,
            "h2h_home_wins_at_home": 0,
            "h2h_home_losses_away": 0,
        }

    features = {}
    features["h2h_matches"] = len(h2h)

    home_as_home = h2h[(h2h["home_team"] == home_team) & (h2h["result"] == "H")]
    home_as_away = h2h[(h2h["away_team"] == home_team) & (h2h["result"] == "A")]
    features["h2h_home_wins"] = len(home_as_home) + len(home_as_away)

    away_as_home = h2h[(h2h["home_team"] == away_team) & (h2h["result"] == "H")]
    away_as_away = h2h[(h2h["away_team"] == away_team) & (h2h["result"] == "A")]
    features["h2h_away_wins"] = len(away_as_home) + len(away_as_away)

    features["h2h_draws"] = (h2h["result"] == "D").sum()

    home_as_home_df = h2h[h2h["home_team"] == home_team]
    home_as_away_df = h2h[h2h["away_team"] == home_team]

    if not home_as_home_df.empty:
        features["h2h_home_goals_for"] = home_as_home_df["home_goals"].mean()
        features["h2h_home_goals_against"] = home_as_home_df["away_goals"].mean()
        features["h2h_home_wins_at_home"] = (home_as_home_df["result"] == "H").sum()
    else:
        features["h2h_home_goals_for"] = 1.5
        features["h2h_home_goals_against"] = 1.0
        features["h2h_home_wins_at_home"] = 0

    if not home_as_away_df.empty:
        features["h2h_away_goals_for"] = home_as_away_df["away_goals"].mean()
        features["h2h_away_goals_against"] = home_as_away_df["home_goals"].mean()
        features["h2h_home_losses_away"] = (home_as_away_df["result"] == "H").sum()
    else:
        features["h2h_away_goals_for"] = 1.0
        features["h2h_away_goals_against"] = 1.5
        features["h2h_home_losses_away"] = 0

    if "total_goals" in h2h.columns:
        features["h2h_avg_total_goals"] = h2h["total_goals"].mean()
    else:
        features["h2h_avg_total_goals"] = 2.5

    return features


# ═══════════════════════════════════════════════════════
# SECTION 2 — BUILD FEATURE VECTOR
# ═══════════════════════════════════════════════════════

def build_current_features(
    home_team: str,
    away_team: str,
    home_shots: float,
    away_shots: float,
    home_corners: float,
    away_corners: float,
    features_df: pd.DataFrame,
    league_key: str = "EPL",
    home_possession: Optional[float] = None,
    away_possession: Optional[float] = None,
    model_features: Optional[list] = None,
) -> pd.DataFrame:
    """
    Build feature vector using only current match data + H2H.

    Possession is optional — included only when provided AND when the
    saved model was trained with possession features.

    Args:
        home_team: Home team name
        away_team: Away team name
        home_shots: Home team shots in this match
        away_shots: Away team shots in this match
        home_corners: Home team corners in this match
        away_corners: Away team corners in this match
        features_df: Historical data for H2H calculation
        league_key: League code (EPL, LA_LIGA, etc.)
        home_possession: Home team possession % (0-100), optional
        away_possession: Away team possession % (0-100), optional
        model_features: Feature names the model was trained on (from saved .pkl).
                        When provided, X is reindexed to match exactly.

    Returns:
        DataFrame with 1 row containing all features aligned to model_features.
    """
    h2h = get_h2h_features(home_team, away_team, features_df, league_key)

    features_dict = {
        "home_shots": home_shots,
        "away_shots": away_shots,
        "home_corners": home_corners,
        "away_corners": away_corners,
        "shot_diff": home_shots - away_shots,
        "corner_diff": home_corners - away_corners,
    }

    if home_possession is not None and away_possession is not None:
        features_dict["home_possession"] = home_possession
        features_dict["away_possession"] = away_possession
        features_dict["possession_diff"] = home_possession - away_possession

    features_dict.update(h2h)

    X = pd.DataFrame([features_dict])

    if model_features is not None:
        for col in model_features:
            if col not in X.columns:
                X[col] = 0.0
        X = X[model_features]

    return X


# ═══════════════════════════════════════════════════════
# SECTION 3 — LOAD MODEL + FEATURES HELPER
# ═══════════════════════════════════════════════════════

def _load_model_and_features(model_path: Path, features_path: Path):
    """Load a model and its saved feature list."""
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}\n"
            "Run: python src/trainCurrentOnly.py"
        )
    model = joblib.load(model_path)

    features = None
    if features_path.exists():
        features = joblib.load(features_path)
    else:
        logger.warning(f"Feature list not found: {features_path} — column order may mismatch.")

    return model, features


# ═══════════════════════════════════════════════════════
# SECTION 4 — SHAP EXPLANATION (XGBoost multiclass safe)
# ═══════════════════════════════════════════════════════

def _get_shap_values(model, X: pd.DataFrame, top_idx: int) -> list:
    """
    Compute SHAP feature importance for the predicted class.

    Uses shap.Explainer (the newer API) instead of shap.TreeExplainer
    directly, which avoids the XGBoost multiclass base_score vector bug:
        ValueError: could not convert string to float:
        '[2.7812707E-1,-2.358737E-1,-4.2253256E-2]'

    Falls back gracefully if SHAP is not installed or fails.

    Args:
        model: Trained XGBoost / LightGBM model
        X: Feature DataFrame (1 row)
        top_idx: Index of predicted class (0=Home, 1=Draw, 2=Away)

    Returns:
        List of dicts sorted by absolute SHAP value (top 10).
    """
    try:
        import shap
    except ImportError:
        logger.warning("SHAP not installed — pip install shap")
        return []

    try:
        # shap.Explainer auto-selects the right backend and handles
        # XGBoost's multi-class base_score vector correctly.
        explainer = shap.Explainer(model, X)
        shap_values = explainer(X)

        # shap_values.values shape: (n_samples, n_features) for binary,
        # or (n_samples, n_features, n_classes) for multiclass.
        vals = shap_values.values
        if vals.ndim == 3:
            # Multiclass: pick the slice for the predicted class
            vals = vals[0, :, top_idx]
        else:
            vals = vals[0, :]

        feature_importance = [
            {
                "feature": col,
                "shap_value": float(abs(v)),
                "direction": "↑ increases" if v > 0 else "↓ decreases",
            }
            for col, v in zip(X.columns, vals)
        ]
        return sorted(feature_importance, key=lambda x: x["shap_value"], reverse=True)[:10]

    except Exception as e:
        logger.warning(f"SHAP explanation failed: {e}")
        return []


# ═══════════════════════════════════════════════════════
# SECTION 5 — PREDICTION
# ═══════════════════════════════════════════════════════

def predict_match_current_only(
    home_team: str,
    away_team: str,
    home_shots: float,
    away_shots: float,
    home_corners: float,
    away_corners: float,
    features_df: pd.DataFrame,
    league_key: str = "EPL",
    home_possession: Optional[float] = None,
    away_possession: Optional[float] = None,
    include_shap: bool = False
) -> Dict[str, Any]:
    """
    Predict match result using ONLY current data and H2H.

    Args:
        home_team: Home team name
        away_team: Away team name
        home_shots: Home shots this match
        away_shots: Away shots this match
        home_corners: Home corners this match
        away_corners: Away corners this match
        features_df: Historical data for H2H only
        league_key: League code (EPL, LA_LIGA, etc.)
        home_possession: Home possession % (optional)
        away_possession: Away possession % (optional)
        include_shap: Whether to generate SHAP explanations

    Returns:
        Dict with predictions for all markets
    """
    # ── Match result model ──────────────────────────────
    features_pkl = PATHS.MODELS / "match_result_current_features.pkl"
    model, model_features = _load_model_and_features(PATHS.MODEL_MATCH_RESULT, features_pkl)

    X = build_current_features(
        home_team, away_team,
        home_shots, away_shots,
        home_corners, away_corners,
        features_df, league_key,
        home_possession=home_possession,
        away_possession=away_possession,
        model_features=model_features,
    )

    logger.debug(f"Features shape: {X.shape}")
    logger.debug(f"Features columns: {X.columns.tolist()}")

    proba = model.predict_proba(X)[0]
    top_idx = int(np.argmax(proba))
    top_prob = float(proba[top_idx])

    labels = {0: "Home Win", 1: "Draw", 2: "Away Win"}
    confidence = "HIGH" if top_prob >= 0.65 else "MEDIUM" if top_prob >= 0.50 else "LOW"

    result: Dict[str, Any] = {
        "home_team": home_team,
        "away_team": away_team,
        "league_key": league_key,

        "p_home": float(proba[0]),
        "p_draw": float(proba[1]),
        "p_away": float(proba[2]),

        "top_outcome": labels[top_idx],
        "top_probability": round(top_prob, 4),
        "confidence": confidence,

        "match_stats": {
            "home_shots": home_shots,
            "away_shots": away_shots,
            "shot_diff": home_shots - away_shots,
            "home_corners": home_corners,
            "away_corners": away_corners,
            "corner_diff": home_corners - away_corners,
            "home_possession": home_possession,
            "away_possession": away_possession,
        },

        "h2h_info": get_h2h_features(home_team, away_team, features_df, league_key),
    }

    # ── BTTS ────────────────────────────────────────────
    try:
        btts_model, btts_features = _load_model_and_features(
            PATHS.MODEL_BTTS, PATHS.MODELS / "btts_current_features.pkl"
        )
        X_btts = build_current_features(
            home_team, away_team,
            home_shots, away_shots,
            home_corners, away_corners,
            features_df, league_key,
            home_possession=home_possession,
            away_possession=away_possession,
            model_features=btts_features,
        )
        result["p_btts"] = float(btts_model.predict_proba(X_btts)[0][1])
    except FileNotFoundError:
        logger.warning("BTTS model not found — skipping")

    # ── Over/Under 2.5 ──────────────────────────────────
    try:
        ou_model, ou_features = _load_model_and_features(
            PATHS.MODEL_OVER_UNDER, PATHS.MODELS / "over_under_current_features.pkl"
        )
        X_ou = build_current_features(
            home_team, away_team,
            home_shots, away_shots,
            home_corners, away_corners,
            features_df, league_key,
            home_possession=home_possession,
            away_possession=away_possession,
            model_features=ou_features,
        )
        result["p_over_25"] = float(ou_model.predict_proba(X_ou)[0][1])
    except FileNotFoundError:
        logger.warning("Over/Under model not found — skipping")

    # ── Corners ─────────────────────────────────────────
    try:
        corners_model, corners_features = _load_model_and_features(
            PATHS.MODEL_CORNERS, PATHS.MODELS / "corners_current_features.pkl"
        )
        X_corners = build_current_features(
            home_team, away_team,
            home_shots, away_shots,
            home_corners, away_corners,
            features_df, league_key,
            home_possession=home_possession,
            away_possession=away_possession,
            model_features=corners_features,
        )
        result["p_corners_over_95"] = float(corners_model.predict_proba(X_corners)[0][1])
    except FileNotFoundError:
        logger.warning("Corners model not found — skipping")

    # ── SHAP ────────────────────────────────────────────
    if include_shap:
        result["shap_values"] = _get_shap_values(model, X, top_idx)

    return result


# ═══════════════════════════════════════════════════════
# SECTION 6 — MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════

def main():
    """Example usage."""
    try:
        features_df = pd.read_parquet(PATHS.PROCESSED / "master.parquet")
    except FileNotFoundError:
        logger.error("Features not found. Run: python src/update_data.py")
        return

    result = predict_match_current_only(
        home_team="Arsenal",
        away_team="Chelsea",
        home_shots=15,
        away_shots=8,
        home_corners=6,
        away_corners=4,
        features_df=features_df,
        league_key="EPL",
        home_possession=58.5,   # set to None if unavailable
        away_possession=41.5,
        include_shap=True
    )

    print("\n" + "="*60)
    print(f"{result['home_team']} vs {result['away_team']}")
    print("="*60)
    print(f"\nPrediction: {result['top_outcome']} ({result['top_probability']:.1%})")
    print(f"Confidence: {result['confidence']}")
    print(f"\nAll outcomes:")
    print(f"  Home Win: {result['p_home']:.1%}")
    print(f"  Draw:     {result['p_draw']:.1%}")
    print(f"  Away Win: {result['p_away']:.1%}")

    if "p_btts" in result:
        print(f"\nBTTS:         {result['p_btts']:.1%}")
    if "p_over_25" in result:
        print(f"Over 2.5:     {result['p_over_25']:.1%}")
    if "p_corners_over_95" in result:
        print(f"Corners >9.5: {result['p_corners_over_95']:.1%}")

    print(f"\nMatch stats:")
    stats = result["match_stats"]
    print(f"  Shots:   {stats['home_shots']:.0f} - {stats['away_shots']:.0f} (diff: {stats['shot_diff']:+.0f})")
    print(f"  Corners: {stats['home_corners']:.0f} - {stats['away_corners']:.0f} (diff: {stats['corner_diff']:+.0f})")
    if stats["home_possession"] is not None:
        print(f"  Poss:    {stats['home_possession']:.1f}% - {stats['away_possession']:.1f}%")

    print(f"\nH2H History:")
    h2h = result["h2h_info"]
    print(f"  Matches: {h2h['h2h_matches']:.0f}")
    print(f"  Record:  {h2h['h2h_home_wins']:.0f}W - {h2h['h2h_draws']:.0f}D - {h2h['h2h_away_wins']:.0f}L")
    print(f"  Home avg goals: {h2h['h2h_home_goals_for']:.2f} for, {h2h['h2h_home_goals_against']:.2f} against")
    print(f"  Avg total goals H2H: {h2h['h2h_avg_total_goals']:.2f}")

    if result.get("shap_values"):
        print(f"\nTop SHAP features:")
        for item in result["shap_values"][:5]:
            print(f"  {item['feature']:30s} {item['direction']}  ({item['shap_value']:.4f})")


if __name__ == "__main__":
    main()