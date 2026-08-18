"""
predict.py — Oracle Prediction Engine v2
==========================================
UPGRADES vs original:
  ① Uses CalibratedFootballModel (isotonic regression) if saved — better probabilities
  ② Blends XGBoost predictions with Dixon-Coles Poisson probabilities
  ③ Exposes xG, referee, line-movement context in prediction output
  ④ predict_all_markets() now returns dc_blend probabilities
  ⑤ Graceful degradation — works with just XGBoost if upgrades not yet run

Run:
    python src/predict.py --today
    python src/predict.py --today --league EPL
    python src/predict.py --home Arsenal --away Chelsea --league EPL --market all
    python src/predict.py --home Arsenal --away Chelsea --league EPL --dc-only
"""

import sys
import re
import datetime
from pathlib import Path
from typing import Any
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import joblib
from loguru import logger
from config import PATHS, LOGGING, VALUE_BET, MARKETS, LEAGUES

logger.remove()
logger.add(sys.stderr, level=LOGGING.LEVEL, format=LOGGING.FORMAT, colorize=True)

RESULT_LABELS  = {0: "Home Win", 1: "Draw", 2: "Away Win"}
CONFIDENCE_MAP = {
    "HIGH":   "🟢 HIGH  (>65%)",
    "MEDIUM": "🟡 MEDIUM (50-65%)",
    "LOW":    "🔴 LOW   (<50%)",
}

# Blend weight: how much Dixon-Coles contributes to final 1X2 probability.
# 0.0 = pure XGBoost, 1.0 = pure DC, 0.35 = recommended blend
DC_BLEND_WEIGHT = 0.35


# ═══════════════════════════════════════════════════════
# SECTION 1 — MODEL LOADERS
# ═══════════════════════════════════════════════════════

def load_best_model(market: str = "match_result"):
    """
    Load the best available model for a market.

    Priority for match_result:
      1. CalibratedFootballModel (isotonic — best calibrated probabilities)
      2. Base XGBoost model (trained by train.py / update_data.py)

    For other markets: base model only.

    Returns:
        (model, feature_names)
    """
    if market == "match_result":
        cal_path  = PATHS.MODELS / "match_result_calibrated.pkl"
        feat_path = PATHS.MODELS / "match_result_features.pkl"

        if cal_path.exists() and feat_path.exists():
            try:
                try:
                    from src.feature_upgrades import CalibratedFootballModel
                except ImportError:
                    from feature_upgrades import CalibratedFootballModel
                model = CalibratedFootballModel.load(cal_path)
                feats = joblib.load(feat_path)
                logger.debug("Loaded calibrated match_result model ✓")
                return model, feats
            except Exception as e:
                logger.warning(f"Calibrated model failed to load: {e} — using base XGBoost")

    # Fall back to base model
    model_paths = {
        "match_result": (PATHS.MODEL_MATCH_RESULT,
                         PATHS.MODELS / "match_result_features.pkl"),
        "btts":         (PATHS.MODEL_BTTS,
                         PATHS.MODELS / "btts_features.pkl"),
        "over_under":   (PATHS.MODEL_OVER_UNDER,
                         PATHS.MODELS / "over_under_features.pkl"),
        "corners":      (PATHS.MODEL_CORNERS,
                         PATHS.MODELS / "corners_features.pkl"),
    }
    if market not in model_paths:
        raise ValueError(f"Unknown market '{market}'. Choose: {list(model_paths)}")

    mpath, fpath = model_paths[market]
    if not mpath.exists():
        raise FileNotFoundError(f"No model at {mpath}. Run: python src/update_data.py")

    model  = joblib.load(mpath)
    feats  = joblib.load(fpath) if fpath.exists() else None
    logger.debug(f"Loaded {market} base model")
    return model, feats


def load_dixon_coles():
    """
    Load the Dixon-Coles model if available.
    Returns None silently if not built yet (graceful degradation).
    """
    try:
        try:
            from src.dixon_coles import DixonColesModel
        except ImportError:
            from dixon_coles import DixonColesModel
        return DixonColesModel.load()
    except (FileNotFoundError, Exception) as e:
        logger.debug(f"Dixon-Coles model not found ({e}) — using XGBoost only")
        return None


def load_features() -> pd.DataFrame:
    """Load pre-built feature parquet from disk."""
    for fname in ["features_master.parquet", "master_features.parquet",
                  "features.parquet", "master_with_elo.parquet", "master.parquet"]:
        p = PATHS.PROCESSED / fname
        if p.exists():
            logger.debug(f"Loading features: {fname}")
            df = pd.read_parquet(p, engine="auto")
            df["date"] = pd.to_datetime(df["date"])
            return df
    raise FileNotFoundError(
        "No features parquet found.\n"
        "Run: python src/update_data.py"
    )


# ═══════════════════════════════════════════════════════
# SECTION 2 — FEATURE PREPARATION
# ═══════════════════════════════════════════════════════

def sanitise_columns(df: pd.DataFrame) -> pd.DataFrame:
    seen, nc = {}, {}
    for col in df.columns:
        c = re.sub(r'[^a-zA-Z0-9_]', '_', str(col))
        if c in seen:
            seen[c] += 1; c = f"{c}_{seen[c]}"
        else: seen[c] = 0
        nc[col] = c
    return df.rename(columns=nc)


def build_match_row(
    features_df: pd.DataFrame,
    home_team:   str,
    away_team:   str,
    league_key:  str,
    match_date:  str | None = None,
) -> pd.DataFrame:
    """Build a single feature row for prediction from historical data."""
    df = features_df[features_df["league_key"] == league_key].copy()
    if match_date:
        df = df[df["date"] < pd.to_datetime(match_date)]

    # 1. Try most recent H2H
    h2h = df[
        ((df["home_team"] == home_team) & (df["away_team"] == away_team)) |
        ((df["home_team"] == away_team) & (df["away_team"] == home_team))
    ]
    if not h2h.empty:
        row = h2h.sort_values("date").iloc[[-1]].copy()
        if row.iloc[0]["home_team"] != home_team:
            for col in [c for c in row.columns if c.startswith("home_")]:
                ac = col.replace("home_", "away_", 1)
                if ac in row.columns:
                    row[col], row[ac] = row[ac].values.copy(), row[col].values.copy()
            row["home_team"] = home_team
            row["away_team"] = away_team
        return row

    # 2. Stitch from individual team rows
    logger.warning(f"No H2H found for {home_team} vs {away_team}. Stitching rows.")
    hr = df[(df["home_team"] == home_team) | (df["away_team"] == home_team)]
    ar = df[(df["home_team"] == away_team) | (df["away_team"] == away_team)]
    if hr.empty:
        raise ValueError(f"'{home_team}' not found in {league_key}")
    if ar.empty:
        raise ValueError(f"'{away_team}' not found in {league_key}")
    row = hr.sort_values("date").iloc[[-1]].copy()
    al  = ar.sort_values("date").iloc[-1]
    for col in row.columns:
        if col.startswith("away_"):
            row[col] = al.get(col, row[col].values[0])
    row["home_team"] = home_team
    row["away_team"] = away_team
    return row


def prepare_X(row: pd.DataFrame, feature_names: list[str] | None) -> pd.DataFrame:
    """Sanitise and select model features from a match row."""
    if feature_names is None:
        feature_names = []

    row = sanitise_columns(row.copy())
    for col in feature_names:
        if col not in row.columns:
            row[col] = -999.0
    return row[feature_names].astype(np.float32).fillna(-999)


# ═══════════════════════════════════════════════════════
# SECTION 3 — PROBABILITY BLENDING (XGBoost + Dixon-Coles)
# ═══════════════════════════════════════════════════════

def blend_probabilities(
    xgb_proba: np.ndarray,
    dc_probs: dict | None,
    blend_weight: float = DC_BLEND_WEIGHT,
) -> np.ndarray:
    """
    Blend XGBoost and Dixon-Coles probabilities for 1X2.

    XGBoost captures complex feature interactions (form, ELO, xG, referee).
    Dixon-Coles captures the bivariate score distribution (better for BTTS/O-U).
    Blending them gives better calibrated 1X2 probabilities than either alone.

    blend_weight = weight given to Dixon-Coles (0 = pure XGB, 1 = pure DC)

    Args:
        xgb_proba:    np.array [P(H), P(D), P(A)] from XGBoost
        dc_probs:     Dict from DixonColesModel.predict_probs()
        blend_weight: DC contribution (default 0.35)

    Returns:
        Blended np.array [P(H), P(D), P(A)] normalised to sum 1.
    """
    if dc_probs is None or blend_weight == 0:
        return xgb_proba

    dc_arr = np.array([
        dc_probs.get("p_home", xgb_proba[0]),
        dc_probs.get("p_draw", xgb_proba[1]),
        dc_probs.get("p_away", xgb_proba[2]),
    ])
    blended = (1 - blend_weight) * xgb_proba + blend_weight * dc_arr
    return blended / blended.sum()


# ═══════════════════════════════════════════════════════
# SECTION 4 — SINGLE MARKET PREDICTION
# ═══════════════════════════════════════════════════════

def predict_market(
    home_team:   str,
    away_team:   str,
    league_key:  str,
    market:      str = "match_result",
    match_date:  str | None = None,
    features_df: pd.DataFrame | None = None,
    dc_model=None,
    blend_weight: float = DC_BLEND_WEIGHT,
) -> dict:
    """
    Predict one market for one match.

    For match_result: blends XGBoost (calibrated if available) + Dixon-Coles.
    For other markets: uses respective model directly.

    Returns:
        Dict with probabilities, top_outcome, confidence, xg_context, shap_values.
    """
    if features_df is None:
        features_df = load_features()

    model, feature_names = load_best_model(market)
    row = build_match_row(features_df, home_team, away_team, league_key, match_date)
    X   = prepare_X(row, feature_names)

    proba  = model.predict_proba(X)[0]
    dc_ctx = None

    # Blend with Dixon-Coles for match result
    if market == "match_result":
        if dc_model is None:
            dc_model = load_dixon_coles()
        if dc_model is not None:
            try:
                dc_pred = dc_model.predict_probs(home_team, away_team,
                                                  include_scorelines=False)
                proba   = blend_probabilities(proba, dc_pred, blend_weight)
                dc_ctx  = {
                    "lambda_home":          dc_pred.get("lambda_home"),
                    "lambda_away":          dc_pred.get("lambda_away"),
                    "dc_p_btts":            dc_pred.get("p_btts"),
                    "dc_p_over_25":         dc_pred.get("p_over_25"),
                    "dc_p_over_15":         dc_pred.get("p_over_15"),
                    "dc_expected_goals":    dc_pred.get("expected_total_goals"),
                    "blend_weight_dc":      blend_weight,
                }
            except Exception as e:
                logger.debug(f"DC blend failed: {e}")

    # Build labels per market
    if market == "match_result":
        labels    = {RESULT_LABELS[i]: float(proba[i]) for i in range(len(proba))}
        top_idx   = int(np.argmax(proba))
        top_label = RESULT_LABELS[top_idx]

    elif market == "btts":
        labels    = {"Yes": float(proba[1]), "No": float(proba[0])}
        top_label = list(labels.keys())[int(np.argmax(list(labels.values())))]
        top_idx   = 1

    elif market == "over_under":
        labels    = {"Over 2.5": float(proba[1]), "Under 2.5": float(proba[0])}
        top_label = list(labels.keys())[int(np.argmax(list(labels.values())))]
        top_idx   = 1

    elif market == "corners":
        labels    = {"Over 9.5": float(proba[1]), "Under 9.5": float(proba[0])}
        top_label = list(labels.keys())[int(np.argmax(list(labels.values())))]
        top_idx   = 1

    else:
        labels    = {str(i): float(p) for i, p in enumerate(proba)}
        top_label = str(int(np.argmax(proba)))
        top_idx   = 0

    top_prob   = max(labels.values())
    confidence = (
        "HIGH"   if top_prob >= VALUE_BET.HIGH_CONFIDENCE   else
        "MEDIUM" if top_prob >= VALUE_BET.MEDIUM_CONFIDENCE else "LOW"
    )

    # Extract upgrade context from the feature row
    row_san = sanitise_columns(row.copy()).iloc[0]
    xg_ctx  = _extract_xg_context(row_san)
    ref_ctx = _extract_referee_context(row_san)
    lm_ctx  = _extract_line_movement_context(row_san)

    return {
        "home_team":        home_team,
        "away_team":        away_team,
        "league":           league_key,
        "market":           MARKETS.get(market, market),
        "market_key":       market,
        "probabilities":    labels,
        "top_outcome":      top_label,
        "top_probability":  round(top_prob, 4),
        "confidence":       confidence,
        # Upgrade contexts (None if data not yet available)
        "dixon_coles":      dc_ctx,
        "xg_context":       xg_ctx,
        "referee_context":  ref_ctx,
        "line_movement":    lm_ctx,
        # SHAP (slow — only computed in full predict)
        "shap_values":      compute_shap(model, X, top_idx),
        "raw_proba":        proba.tolist(),
    }


# ═══════════════════════════════════════════════════════
# SECTION 5 — ALL MARKETS IN ONE CALL
# ═══════════════════════════════════════════════════════

def predict_all_markets(
    home_team:    str,
    away_team:    str,
    league_key:   str,
    match_date:   str | None = None,
    features_df:  pd.DataFrame | None = None,
    blend_weight: float = DC_BLEND_WEIGHT,
) -> dict:
    """
    Run all available market models + Dixon-Coles for one match.

    Loads Dixon-Coles once and passes it to all market predictions
    (efficient — avoids multiple disk reads).

    Returns:
        {
            home_team, away_team, league,
            markets: {
                match_result: {probabilities, top_outcome, confidence,
                               dixon_coles, xg_context, referee_context,
                               line_movement, shap_values},
                btts:         {...},
                over_under:   {...},
                corners:      {...},
            },
            predictions_flat:  {market: {outcome: prob}},
            model_verdict:     "Home Win @ 58% — MEDIUM confidence",
            overall_confidence:"MEDIUM",
            dc_available:      True/False,
            upgrade_summary:   {xg: bool, dc: bool, referee: bool, line_movement: bool},
            errors:            [],
        }
    """
    if features_df is None:
        features_df = load_features()

    # Load DC once for efficiency
    dc_model = load_dixon_coles()

    markets_to_run = []
    model_map = {
        "match_result": PATHS.MODEL_MATCH_RESULT,
        "btts":         PATHS.MODEL_BTTS,
        "over_under":   PATHS.MODEL_OVER_UNDER,
        "corners":      PATHS.MODEL_CORNERS,
    }
    for mk, mp in model_map.items():
        if mp.exists():
            markets_to_run.append(mk)

    market_results:   dict[str, Any] = {}
    predictions_flat: dict[str, Any] = {}
    errors:           list[str]      = []

    for market in markets_to_run:
        try:
            result = predict_market(
                home_team, away_team, league_key,
                market=market,
                match_date=match_date,
                features_df=features_df,
                dc_model=dc_model,
                blend_weight=blend_weight,
            )
            market_results[market]   = result
            predictions_flat[market] = result["probabilities"]
        except Exception as e:
            errors.append(f"{market}: {e}")
            logger.warning(f"Market {market} failed: {e}")

    # Verdict from match result
    mr = market_results.get("match_result", {})
    if mr:
        top_out  = mr["top_outcome"]
        top_prob = mr["top_probability"]
        conf     = mr["confidence"]
        verdict  = f"{top_out} @ {top_prob:.0%} — {conf} confidence"
        if mr.get("dixon_coles"):
            dc = mr["dixon_coles"]
            verdict += (f"  |  xG: {dc['lambda_home']:.2f}-{dc['lambda_away']:.2f}"
                        f"  |  BTTS: {dc['dc_p_btts']:.0%}"
                        f"  |  O2.5: {dc['dc_p_over_25']:.0%}")
        overall_conf = conf
    else:
        verdict = "Model unavailable"; overall_conf = "LOW"

    # Upgrade availability summary
    def _has_xg(r: dict | None) -> bool:
        ctx = r.get("xg_context", {}) if r else {}
        return bool(ctx and ctx.get("home_xg_roll5") is not None)

    def _has_ref(r: dict | None) -> bool:
        ctx = r.get("referee_context", {}) if r else {}
        return bool(ctx and ctx.get("ref_known") == 1.0)

    def _has_lm(r: dict | None) -> bool:
        ctx = r.get("line_movement", {}) if r else {}
        return bool(ctx and ctx.get("line_move_home") is not None)

    mr_res = market_results.get("match_result")
    upgrade_summary = {
        "xg":           _has_xg(mr_res),
        "dixon_coles":  dc_model is not None,
        "referee":      _has_ref(mr_res),
        "line_movement":_has_lm(mr_res),
    }

    return {
        "home_team":          home_team,
        "away_team":          away_team,
        "league":             league_key,
        "markets":            market_results,
        "predictions_flat":   predictions_flat,
        "model_verdict":      verdict,
        "overall_confidence": overall_conf,
        "dc_available":       dc_model is not None,
        "upgrade_summary":    upgrade_summary,
        "errors":             errors,
        "markets_run":        list(market_results.keys()),
    }


# ═══════════════════════════════════════════════════════
# SECTION 6 — UPGRADE CONTEXT EXTRACTORS
# ═══════════════════════════════════════════════════════

def _extract_xg_context(row: pd.Series) -> dict | None:
    """Extract xG feature values from a feature row."""
    keys = ["home_xg_roll5", "away_xg_roll5", "home_xga_roll5", "away_xga_roll5",
            "xg_diff_roll5", "home_xg_form_ratio", "away_xg_form_ratio"]
    ctx  = {k: float(row[k]) for k in keys if k in row.index and pd.notna(row[k])}
    return ctx if ctx else None


def _extract_referee_context(row: pd.Series) -> dict | None:
    """Extract referee feature values from a feature row."""
    keys = ["ref_avg_cards", "ref_avg_corners", "ref_home_win_rate",
            "ref_n_games", "ref_known"]
    ctx  = {k: float(row[k]) for k in keys if k in row.index and pd.notna(row[k])}
    return ctx if ctx else None


def _extract_line_movement_context(row: pd.Series) -> dict | None:
    """Extract line movement feature values from a feature row."""
    keys = ["line_move_home", "line_move_draw", "line_move_away",
            "sharp_home_flag", "sharp_away_flag", "home_prob_drift",
            "overround_open", "overround_close"]
    ctx  = {k: float(row[k]) for k in keys if k in row.index and pd.notna(row[k])}
    return ctx if ctx else None


# ═══════════════════════════════════════════════════════
# SECTION 7 — SHAP
# ═══════════════════════════════════════════════════════

def compute_shap(model: Any, X: pd.DataFrame, class_idx: int = 0, top_n: int = 5) -> list[dict[str, Any]]:
    """
    Compute SHAP feature importance for the predicted class.

    Converts the SHAP ndarray to a plain Python list[float] immediately after
    extraction — this sidesteps all numpy Scalar / complex type ambiguity that
    Pylance raises when indexing raw numpy arrays with numpy integer indices.
    """
    try:
        # Unwrap CalibratedFootballModel / sklearn wrapper to reach the raw tree model
        inner: Any = getattr(model, "calibrated_", model)
        inner      = getattr(inner, "estimator",   inner)

        import shap
        explainer = shap.TreeExplainer(inner)
        sv_raw: Any = explainer.shap_values(X)

        # Normalise to a 1-D ndarray for the target class
        sv_nd: np.ndarray = (
            sv_raw[min(class_idx, len(sv_raw) - 1)][0]
            if isinstance(sv_raw, list)
            else sv_raw[0]
        )

        # Convert to plain Python list[float] BEFORE any indexing.
        # This is the key fix: once values are native Python floats,
        # Pylance has no type complaints about any subsequent operations.
        sv_list: list[float] = [float(v) for v in sv_nd.tolist()]
        abs_list: list[float] = [abs(v) for v in sv_list]

        # Convert X row to plain Python floats — avoids Pylance Scalar on iloc indexing
        x_vals: list[float] = [float(v) for v in X.iloc[0].tolist()]

        # Sort by absolute impact, take top_n — pure Python, no numpy indexing
        order: list[int] = sorted(
            range(len(abs_list)), key=lambda k: abs_list[k], reverse=True
        )[:top_n]

        result: list[dict[str, Any]] = []
        for idx in order:
            shap_val: float = sv_list[idx]
            feat_val: float = x_vals[idx]
            result.append({
                "feature":       str(X.columns[idx]),
                "feature_value": round(feat_val, 4),
                "shap_impact":   round(shap_val, 4),
                "direction":     "↑ increases" if shap_val > 0 else "↓ decreases",
            })
        return result

    except Exception as e:
        logger.debug(f"SHAP skipped: {e}")
        return []


# ═══════════════════════════════════════════════════════
# SECTION 8 — TODAY'S FIXTURES
# ═══════════════════════════════════════════════════════

FD_ORG_COMP = {
    "EPL":            "PL",
    "ENG_CHAMP":      "ELC",
    "LA_LIGA":        "PD",
    "SERIE_A":        "SA",
    "LIGUE_1":        "FL1",
    "BUNDESLIGA":     "BL1",
}

TEAM_NAME_MAP = {
    "Arsenal FC":"Arsenal","Chelsea FC":"Chelsea","Liverpool FC":"Liverpool",
    "Manchester City FC":"Man City","Manchester United FC":"Man United",
    "Tottenham Hotspur FC":"Tottenham","Newcastle United FC":"Newcastle",
    "Aston Villa FC":"Aston Villa","West Ham United FC":"West Ham",
    "Brighton & Hove Albion FC":"Brighton","Brentford FC":"Brentford",
    "Fulham FC":"Fulham","Crystal Palace FC":"Crystal Palace",
    "Wolverhampton Wanderers FC":"Wolves","Everton FC":"Everton",
    "Nottingham Forest FC":"Nott'm Forest","AFC Bournemouth":"Bournemouth",
    "Leicester City FC":"Leicester","Ipswich Town FC":"Ipswich",
    "Southampton FC":"Southampton","FC Barcelona":"Barcelona",
    "Real Madrid CF":"Real Madrid","Club Atlético de Madrid":"Ath Madrid",
    "Athletic Club":"Ath Bilbao",
    "Real Sociedad de Futbol":"Sociedad","Real Sociedad de Fútbol":"Sociedad",
    "Villarreal CF":"Villarreal","Sevilla FC":"Sevilla",
    "Real Betis Balompie":"Betis","Real Betis Balompié":"Betis",
    "Rayo Vallecano de Madrid":"Vallecano","Getafe CF":"Getafe",
    "RC Celta de Vigo":"Celta","CA Osasuna":"Osasuna",
    "Deportivo Alaves":"Alaves","Deportivo Alavés":"Alaves",
    "Valencia CF":"Valencia","UD Las Palmas":"Las Palmas","RCD Mallorca":"Mallorca",
    "Girona FC":"Girona","CD Leganes":"Leganes","CD Leganés":"Leganes",
    "Real Valladolid CF":"Valladolid",
    "Espanyol de Barcelona":"Espanol","Cádiz CF":"Cadiz",
    "UD Almería":"Almeria","Real Granada CF":"Granada","Levante UD":"Levante",
    "Inter Milan":"Inter","AC Milan":"Milan","Juventus FC":"Juventus",
    "SSC Napoli":"Napoli","Paris Saint-Germain FC":"Paris SG",
    "FC Bayern München":"Bayern Munich","Borussia Dortmund":"Dortmund",
    "Bayer 04 Leverkusen":"Leverkusen",
    "FC Kobenhavn":"Copenhagen","FC Copenhagen":"Copenhagen",
}


def fetch_todays_fixtures(
    api_key: str | None = None,
    league_key: str | None = None,
) -> tuple[list, list]:
    """Fetch today's fixtures from football-data.org."""
    import requests, time, os
    if not api_key:
        api_key = os.environ.get("FOOTBALL_DATA_KEY", "").strip()
    today    = datetime.datetime.now(datetime.timezone.utc).date().isoformat()
    leagues  = ({league_key: FD_ORG_COMP[league_key]}
                if league_key and league_key in FD_ORG_COMP else FD_ORG_COMP)
    headers  = {"X-Auth-Token": api_key} if api_key else {}
    fixtures: list[dict[str, Any]] = []
    errors:   list[str]            = []
    for lk, code in leagues.items():
        try:
            time.sleep(1.2)
            url  = (f"https://api.football-data.org/v4/competitions/{code}/matches"
                    f"?dateFrom={today}&dateTo={today}")
            resp = requests.get(url, headers=headers, timeout=12)
            if resp.status_code == 200:
                for m in resp.json().get("matches", []):
                    ht = TEAM_NAME_MAP.get(m["homeTeam"]["name"], m["homeTeam"]["name"])
                    at = TEAM_NAME_MAP.get(m["awayTeam"]["name"], m["awayTeam"]["name"])
                    fixtures.append({
                        "home_team": ht, "away_team": at, "league_key": lk,
                        "kickoff":   m.get("utcDate","")[:16],
                        "status":    m.get("status","SCHEDULED"),
                        "matchday":  m.get("matchday",""),
                    })
            elif resp.status_code == 403:
                errors.append(f"{lk}: Not available on your plan")
            elif resp.status_code == 404:
                errors.append(f"{lk}: Competition code not found")
            elif resp.status_code == 429:
                errors.append(f"{lk}: Rate limited")
            else:
                errors.append(f"{lk}: HTTP {resp.status_code}")
        except Exception as e:
            errors.append(f"{lk}: {e}")
    return fixtures, errors


def predict_today(
    api_key: str | None = None,
    league_key: str | None = None,
    blend_weight: float = DC_BLEND_WEIGHT,
) -> list:
    """Predict all today's fixtures. Returns list sorted HIGH→LOW confidence."""
    today        = datetime.datetime.now(datetime.timezone.utc).date().isoformat()
    fixtures, _  = fetch_todays_fixtures(api_key=api_key, league_key=league_key)
    if not fixtures:
        logger.warning(f"No fixtures today ({today})")
        return []
    features_df = load_features()
    dc_model    = load_dixon_coles()
    results: list[dict[str, Any]] = []
    for fix in fixtures:
        try:
            pred = predict_market(
                fix["home_team"], fix["away_team"], fix["league_key"],
                market="match_result", match_date=today,
                features_df=features_df, dc_model=dc_model,
                blend_weight=blend_weight,
            )
            pred["kickoff"]  = fix.get("kickoff", "")
            pred["status"]   = fix.get("status", "")
            pred["matchday"] = fix.get("matchday", "")
            results.append(pred)
        except Exception as e:
            logger.error(f"✗ {fix['home_team']} vs {fix['away_team']}: {e}")
    results.sort(key=lambda x: (
        {"HIGH":0,"MEDIUM":1,"LOW":2}.get(x.get("confidence","LOW"), 2),
        -x.get("top_probability", 0)
    ))
    return results


# ═══════════════════════════════════════════════════════
# SECTION 9 — PRINT HELPERS
# ═══════════════════════════════════════════════════════

def print_prediction(pred: dict, compact: bool = False) -> None:
    if "error" in pred:
        print(f"  ✗ Error: {pred['error']}")
        return
    icon = {"HIGH":"🟢","MEDIUM":"🟡","LOW":"🔴"}.get(pred["confidence"],"⚪")
    if compact:
        probs = pred["probabilities"]
        vals  = list(probs.values())
        keys  = list(probs.keys())
        print(f"  {icon} {pred['home_team']:<22} vs {pred['away_team']:<22} "
              f"| {pred['league']:<10} | "
              f"{keys[0]}:{vals[0]:.0%}  {keys[1]}:{vals[1]:.0%}  {keys[2]}:{vals[2]:.0%}"
              f"  → {pred['top_outcome']} ({pred['top_probability']:.0%})")
    else:
        print(f"\n{'═'*60}")
        print(f"  {pred['home_team']}  vs  {pred['away_team']}")
        print(f"  {pred['league']}  |  {pred['market']}")
        print(f"{'═'*60}")
        for outcome, prob in pred["probabilities"].items():
            print(f"  {outcome:<15} {prob:.1%}  {'█'*int(prob*30)}")
        print(f"\n  TOP PICK:   {pred['top_outcome']}  ({pred['top_probability']:.1%})")
        print(f"  CONFIDENCE: {icon} {CONFIDENCE_MAP.get(pred['confidence'])}")

        # Upgrade outputs
        if pred.get("dixon_coles"):
            dc = pred["dixon_coles"]
            print(f"\n  DIXON-COLES (blend {dc['blend_weight_dc']:.0%}):")
            print(f"    xG: {dc['lambda_home']:.2f} - {dc['lambda_away']:.2f}  "
                  f"| BTTS: {dc['dc_p_btts']:.0%}  "
                  f"| O2.5: {dc['dc_p_over_25']:.0%}  "
                  f"| O1.5: {dc['dc_p_over_15']:.0%}")

        if pred.get("xg_context"):
            xg = pred["xg_context"]
            print(f"\n  xG CONTEXT (rolling 5):")
            print(f"    Home xG for: {xg.get('home_xg_roll5', '?'):.2f}  "
                  f"against: {xg.get('home_xga_roll5', '?'):.2f}  "
                  f"form ratio: {xg.get('home_xg_form_ratio', '?'):.2f}")
            print(f"    Away xG for: {xg.get('away_xg_roll5', '?'):.2f}  "
                  f"against: {xg.get('away_xga_roll5', '?'):.2f}  "
                  f"xG diff: {xg.get('xg_diff_roll5', '?'):+.2f}")

        if pred.get("referee_context"):
            ref = pred["referee_context"]
            known = "✓" if ref.get("ref_known") == 1.0 else "—"
            print(f"\n  REFEREE {known}:")
            print(f"    Avg cards: {ref.get('ref_avg_cards', '?'):.1f}  "
                  f"corners: {ref.get('ref_avg_corners', '?'):.1f}  "
                  f"home win %: {ref.get('ref_home_win_rate', '?'):.0%}")

        if pred.get("line_movement"):
            lm = pred["line_movement"]
            sharp_h = "🚨 SHARP HOME" if lm.get("sharp_home_flag") == 1.0 else ""
            sharp_a = "🚨 SHARP AWAY" if lm.get("sharp_away_flag") == 1.0 else ""
            print(f"\n  LINE MOVEMENT: {sharp_h}{sharp_a}")
            print(f"    Home: {lm.get('line_move_home', 0):+.2f}  "
                  f"Draw: {lm.get('line_move_draw', 0):+.2f}  "
                  f"Away: {lm.get('line_move_away', 0):+.2f}  "
                  f"prob drift: {lm.get('home_prob_drift', 0):+.3f}")

        if pred.get("shap_values"):
            print("\n  TOP FEATURES (SHAP):")
            for i, s in enumerate(pred["shap_values"], 1):
                print(f"    {i}. {s['feature']:<32} "
                      f"val={s['feature_value']:>8.3f}  "
                      f"impact={s['shap_impact']:>+.4f}  {s['direction']}")
        print(f"{'═'*60}")


def print_today_summary(predictions: list) -> None:
    today = datetime.datetime.now(datetime.timezone.utc).date().isoformat()
    print(f"\n{'═'*90}")
    print(f"  TODAY'S PREDICTIONS  —  {today}  [v2: upgrades active]")
    print(f"{'═'*90}")
    print(f"  {'CONF':<8} {'HOME':<22} {'AWAY':<22} {'LGE':<10} "
          f"{'TOP PICK':<14} {'PROB':>5}  {'BTTS':>5}  {'O2.5':>5}")
    print(f"  {'─'*85}")
    for pred in predictions:
        icon  = {"HIGH":"🟢 HIGH","MEDIUM":"🟡 MED","LOW":"🔴 LOW"}.get(
                 pred["confidence"],"⚪")
        btts  = "–"
        over  = "–"
        dc    = pred.get("dixon_coles")
        if dc:
            btts = f"{dc.get('dc_p_btts',0):.0%}"
            over = f"{dc.get('dc_p_over_25',0):.0%}"
        print(f"  {icon:<10} {pred['home_team']:<22} {pred['away_team']:<22} "
              f"{pred['league']:<10} {pred['top_outcome']:<14} "
              f"{pred['top_probability']:>5.1%}  {btts:>5}  {over:>5}")
    print(f"{'═'*90}\n")


# ═══════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse, os

    parser = argparse.ArgumentParser(description="Oracle Prediction Engine v2")
    parser.add_argument("--home",        default=None)
    parser.add_argument("--away",        default=None)
    parser.add_argument("--league",      default=None)
    parser.add_argument("--market",      default="match_result",
                        choices=["match_result","btts","over_under","corners","all"])
    parser.add_argument("--date",        default=None)
    parser.add_argument("--today",       action="store_true")
    parser.add_argument("--dc-only",     action="store_true",
                        help="Use Dixon-Coles only (no XGBoost blend)")
    parser.add_argument("--no-dc",       action="store_true",
                        help="Disable Dixon-Coles blending")
    parser.add_argument("--blend",       type=float, default=DC_BLEND_WEIGHT,
                        help=f"DC blend weight 0-1 (default {DC_BLEND_WEIGHT})")
    args = parser.parse_args()

    PATHS.create_all()
    api_key      = os.environ.get("FOOTBALL_DATA_KEY", "").strip()
    blend_weight = 1.0 if args.dc_only else (0.0 if args.no_dc else args.blend)

    if args.today:
        preds = predict_today(api_key=api_key, league_key=args.league,
                              blend_weight=blend_weight)
        if preds:
            print_today_summary(preds)
            print("\n📊 Full detail — most confident pick:")
            print_prediction(preds[0], compact=False)
        else:
            print("No predictions generated for today.")

    elif args.home and args.away and args.league:
        if args.market == "all":
            result = predict_all_markets(
                args.home, args.away, args.league,
                match_date=args.date, blend_weight=blend_weight,
            )
            print(f"\n{'═'*60}")
            print(f"  {result['home_team']} vs {result['away_team']} — {result['league']}")
            print(f"  Verdict: {result['model_verdict']}")
            print(f"  Upgrades: {result['upgrade_summary']}")
            print(f"{'═'*60}")
            for mkey, mresult in result["markets"].items():
                print(f"\n  [{mresult['market']}]")
                for outcome, prob in mresult["probabilities"].items():
                    print(f"    {outcome:<15} {prob:.1%}  {'█'*int(prob*28)}")
        else:
            pred = predict_market(
                args.home, args.away, args.league,
                market=args.market, match_date=args.date,
                blend_weight=blend_weight,
            )
            print_prediction(pred, compact=False)
    else:
        parser.print_help()
        print("\nExamples:")
        print("  python src/predict.py --today")
        print("  python src/predict.py --today --league EPL")
        print("  python src/predict.py --home Arsenal --away Chelsea --league EPL --market all")
        print("  python src/predict.py --home Arsenal --away Chelsea --league EPL --dc-only")