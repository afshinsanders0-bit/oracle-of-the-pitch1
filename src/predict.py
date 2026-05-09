"""
predict.py — Generate Predictions with SHAP Explanations
==========================================================
UPGRADES:
    - predict_all_markets(): runs all 4 market models in one call
      and returns combined output ready for BetBuilder
    - load_features() correctly calls load_features_df() (reads parquet)
    - --today flag fetches live fixtures from football-data.org API

Run:
    python src/predict.py --today
    python src/predict.py --today --league EPL
    python src/predict.py --home Arsenal --away Chelsea --league EPL
"""
import sys, re, datetime
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import joblib
from loguru import logger
from config import PATHS, LOGGING, VALUE_BET, MARKETS

logger.remove()
logger.add(sys.stderr, level=LOGGING.LEVEL, format=LOGGING.FORMAT, colorize=True)

RESULT_LABELS  = {0: "Home Win", 1: "Draw", 2: "Away Win"}
CONFIDENCE_MAP = {
    "HIGH":   "🟢 HIGH (>65%)",
    "MEDIUM": "🟡 MEDIUM (50-65%)",
    "LOW":    "🔴 LOW (<50%)",
}


# ═══════════════════════════════════════════════════════
# SECTION 1 — DATA LOADERS
# ═══════════════════════════════════════════════════════

def load_features() -> pd.DataFrame:
    """
    Load pre-built feature parquet from disk.
    Tries multiple filenames for compatibility with different pipeline runs.
    Does NOT rebuild features — just reads from disk.
    """
    for fname in [
        "master_features.parquet",
        "features_master.parquet",
        "features.parquet",
        "master_with_elo.parquet",
        "master.parquet",
    ]:
        p = PATHS.PROCESSED / fname
        if p.exists():
            logger.info(f"Loading features: {fname}")
            df = pd.read_parquet(p, engine="pyarrow")
            df["date"] = pd.to_datetime(df["date"])
            return df
    raise FileNotFoundError(
        "No features parquet found in data/processed/\n"
        "Run: python src/data_loader.py && python src/feature_engineering.py"
    )


def load_model(market: str = "match_result"):
    """Load a trained model and its saved feature list from disk."""
    model_paths = {
        "match_result": (PATHS.MODEL_MATCH_RESULT, PATHS.MODELS / "match_result_features.pkl"),
        "btts":         (PATHS.MODEL_BTTS,          PATHS.MODELS / "btts_features.pkl"),
        "over_under":   (PATHS.MODEL_OVER_UNDER,    PATHS.MODELS / "over_under_features.pkl"),
        "corners":      (PATHS.MODEL_CORNERS,       PATHS.MODELS / "corners_features.pkl"),
    }
    if market not in model_paths:
        raise ValueError(f"Unknown market '{market}'. Choose: {list(model_paths)}")

    mpath, fpath = model_paths[market]
    if not mpath.exists():
        raise FileNotFoundError(f"No model at {mpath}. Run: python src/train.py")

    model    = joblib.load(mpath)
    features = joblib.load(fpath) if fpath.exists() else None
    logger.debug(f"Loaded {market} model")
    return model, features


# ═══════════════════════════════════════════════════════
# SECTION 2 — FEATURE PREPARATION
# ═══════════════════════════════════════════════════════

def sanitise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove characters XGBoost rejects from column names."""
    seen, new_cols = {}, {}
    for col in df.columns:
        clean = re.sub(r'[^a-zA-Z0-9_]', '_', str(col))
        if clean in seen:
            seen[clean] += 1
            clean = f"{clean}_{seen[clean]}"
        else:
            seen[clean] = 0
        new_cols[col] = clean
    return df.rename(columns=new_cols)


def build_match_row(
    features_df: pd.DataFrame,
    home_team:   str,
    away_team:   str,
    league_key:  str,
    match_date:  str | None = None,
) -> pd.DataFrame:
    """
    Build a feature row for an upcoming match from historical data.

    Priority order:
    1. Most recent H2H match between the two teams (swap columns if needed)
    2. Stitch from most recent home row + most recent away row

    Args:
        features_df: Historical feature DataFrame
        home_team:   Home team name
        away_team:   Away team name
        league_key:  League key e.g. 'EPL'
        match_date:  ISO date string — only use data before this date

    Returns:
        Single-row DataFrame ready for model.predict_proba()
    """
    df = features_df[features_df["league_key"] == league_key].copy()
    if match_date:
        df = df[df["date"] < pd.to_datetime(match_date)]

    # ── Option 1: Use most recent H2H match ───────────────────────────────
    h2h = df[
        ((df["home_team"] == home_team) & (df["away_team"] == away_team)) |
        ((df["home_team"] == away_team) & (df["away_team"] == home_team))
    ]

    if not h2h.empty:
        row = h2h.sort_values("date").iloc[[-1]].copy()
        # If teams were reversed in the H2H match, swap home/away feature columns
        if row.iloc[0]["home_team"] != home_team:
            for col in [c for c in row.columns if c.startswith("home_")]:
                away_col = col.replace("home_", "away_", 1)
                if away_col in row.columns:
                    row[col], row[away_col] = (
                        row[away_col].values.copy(),
                        row[col].values.copy(),
                    )
            row["home_team"] = home_team
            row["away_team"] = away_team
        return row

    # ── Option 2: Stitch from individual team rows ────────────────────────
    logger.warning(f"No H2H: {home_team} vs {away_team}. Stitching from individual rows.")
    hr = df[df["home_team"] == home_team]
    ar = df[df["away_team"] == away_team]

    if hr.empty:
        raise ValueError(f"'{home_team}' not found in {league_key}. Check spelling.")
    if ar.empty:
        raise ValueError(f"'{away_team}' not found in {league_key}. Check spelling.")

    row = hr.sort_values("date").iloc[[-1]].copy()
    al  = ar.sort_values("date").iloc[-1]
    for col in row.columns:
        if col.startswith("away_"):
            row[col] = al.get(col, row[col].values[0])
    row["home_team"] = home_team
    row["away_team"] = away_team
    return row


def prepare_X(row: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    """Sanitise and select model features from a match row."""
    row = sanitise_columns(row.copy())
    for col in feature_names:
        if col not in row.columns:
            row[col] = -999.0
    return row[feature_names].astype(np.float32).fillna(-999)


# ═══════════════════════════════════════════════════════
# SECTION 3 — SINGLE MARKET PREDICTION
# ═══════════════════════════════════════════════════════

def predict_market(
    home_team:   str,
    away_team:   str,
    league_key:  str,
    market:      str = "match_result",
    match_date:  str | None = None,
    features_df: pd.DataFrame | None = None,
) -> dict:
    """
    Predict one market for one match.

    Returns dict with probabilities, top outcome, confidence, and SHAP values.
    """
    if features_df is None:
        features_df = load_features()

    model, feature_names = load_model(market)
    row = build_match_row(features_df, home_team, away_team, league_key, match_date)

    X     = prepare_X(row, feature_names)
    proba = model.predict_proba(X)[0]

    if market == "match_result":
        labels    = {RESULT_LABELS[i]: float(proba[i]) for i in range(len(proba))}
        top_idx   = int(np.argmax(proba))
        top_label = RESULT_LABELS[top_idx]

    elif market == "btts":
        labels    = {"Yes": float(proba[1]), "No": float(proba[0])}
        top_label = max(labels, key=labels.get)
        top_idx   = 1

    elif market == "over_under":
        labels    = {"Over 2.5": float(proba[1]), "Under 2.5": float(proba[0])}
        top_label = max(labels, key=labels.get)
        top_idx   = 1

    elif market == "corners":
        labels    = {"Over 9.5": float(proba[1]), "Under 9.5": float(proba[0])}
        top_label = max(labels, key=labels.get)
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

    return {
        "home_team":      home_team,
        "away_team":      away_team,
        "league":         league_key,
        "market":         MARKETS.get(market, market),
        "market_key":     market,
        "probabilities":  labels,
        "top_outcome":    top_label,
        "top_probability":round(top_prob, 4),
        "confidence":     confidence,
        "shap_values":    compute_shap(model, X, top_idx),
        "raw_proba":      proba,
    }


# ═══════════════════════════════════════════════════════
# SECTION 4 — ALL MARKETS IN ONE CALL (bet builder feed)
# ═══════════════════════════════════════════════════════

def predict_all_markets(
    home_team:   str,
    away_team:   str,
    league_key:  str,
    match_date:  str | None = None,
    features_df: pd.DataFrame | None = None,
) -> dict:
    """
    Run all available market models for one match and return combined output.

    This is the primary function used by:
    - The Match Deep Dive page (show all market predictions)
    - The BetBuilder (build accumulators across markets)

    Args:
        home_team:   Home team name
        away_team:   Away team name
        league_key:  League key e.g. 'EPL'
        match_date:  ISO date — only use historical data before this
        features_df: Pre-loaded features DataFrame (loaded once for efficiency)

    Returns:
        {
            "home_team": ..., "away_team": ..., "league": ...,
            "markets": {
                "match_result": {"probabilities": {...}, "top_outcome": ..., ...},
                "btts":         {"probabilities": {...}, ...},
                "over_under":   {"probabilities": {...}, ...},
                "corners":      {"probabilities": {...}, ...},
            },
            # Flat view for BetBuilder
            "predictions_flat": {
                "match_result": {"Home Win": 0.52, "Draw": 0.27, "Away Win": 0.21},
                "btts":         {"Yes": 0.61, "No": 0.39},
                ...
            },
            "model_verdict": "Home Win @ 52% — MEDIUM confidence",
            "overall_confidence": "MEDIUM",
        }
    """
    if features_df is None:
        features_df = load_features()

    markets_available = []
    for market in ["match_result", "btts", "over_under", "corners"]:
        mp = {
            "match_result": PATHS.MODEL_MATCH_RESULT,
            "btts":         PATHS.MODEL_BTTS,
            "over_under":   PATHS.MODEL_OVER_UNDER,
            "corners":      PATHS.MODEL_CORNERS,
        }[market]
        if mp.exists():
            markets_available.append(market)

    market_results   = {}
    predictions_flat = {}
    errors           = []

    for market in markets_available:
        try:
            result = predict_market(
                home_team, away_team, league_key,
                market=market,
                match_date=match_date,
                features_df=features_df,
            )
            market_results[market]   = result
            predictions_flat[market] = result["probabilities"]
        except Exception as e:
            errors.append(f"{market}: {e}")
            logger.warning(f"Market {market} failed: {e}")

    # Overall model verdict from match result
    mr = market_results.get("match_result", {})
    if mr:
        top_out  = mr["top_outcome"]
        top_prob = mr["top_probability"]
        conf     = mr["confidence"]
        verdict  = f"{top_out} @ {top_prob:.0%} — {conf} confidence"
        overall_conf = conf
    else:
        verdict      = "Model unavailable"
        overall_conf = "LOW"

    return {
        "home_team":         home_team,
        "away_team":         away_team,
        "league":            league_key,
        "markets":           market_results,
        "predictions_flat":  predictions_flat,
        "model_verdict":     verdict,
        "overall_confidence":overall_conf,
        "errors":            errors,
        "markets_run":       list(market_results.keys()),
    }


# ═══════════════════════════════════════════════════════
# SECTION 5 — SHAP EXPLANATION
# ═══════════════════════════════════════════════════════

def compute_shap(model, X: pd.DataFrame, class_idx: int = 0, top_n: int = 5) -> list[dict]:
    """Compute SHAP values and return top N driving features."""
    try:
        import shap
        explainer = shap.TreeExplainer(model)
        sv        = explainer.shap_values(X)
        sv        = sv[min(class_idx, len(sv)-1)][0] if isinstance(sv, list) else sv[0]
        order     = np.argsort(np.abs(sv))[::-1][:top_n]
        return [
            {
                "feature":       X.columns[i],
                "feature_value": round(float(X.iloc[0, i]), 4),
                "shap_impact":   round(float(sv[i]), 4),
                "direction":     "↑ increases" if sv[i] > 0 else "↓ decreases",
            }
            for i in order
        ]
    except Exception as e:
        logger.warning(f"SHAP skipped: {e}")
        return []


# ═══════════════════════════════════════════════════════
# SECTION 6 — TODAY'S FIXTURES
# ═══════════════════════════════════════════════════════

FD_ORG_COMP = {
    "EPL":     "PL",
    "LA_LIGA": "PD",
    "SERIE_A": "SA",
    "LIGUE_1": "FL1",
    "ENG_CHAMP": "ELC",
}

FD_ORG_TO_PARQUET = {
    "Arsenal FC":"Arsenal","Chelsea FC":"Chelsea","Liverpool FC":"Liverpool",
    "Manchester City FC":"Man City","Manchester United FC":"Man United",
    "Tottenham Hotspur FC":"Tottenham","Newcastle United FC":"Newcastle",
    "Aston Villa FC":"Aston Villa","West Ham United FC":"West Ham",
    "Brighton & Hove Albion FC":"Brighton","Brentford FC":"Brentford",
    "Fulham FC":"Fulham","Crystal Palace FC":"Crystal Palace",
    "Wolverhampton Wanderers FC":"Wolves","Everton FC":"Everton",
    "Nottingham Forest FC":"Nott'm Forest","AFC Bournemouth":"Bournemouth",
    "Leicester City FC":"Leicester","Ipswich Town FC":"Ipswich",
    "Southampton FC":"Southampton",
    "FC Barcelona":"Barcelona","Real Madrid CF":"Real Madrid",
    "Club Atlético de Madrid":"Ath Madrid","Athletic Club":"Ath Bilbao",
    "Real Sociedad de Fútbol":"Sociedad","Villarreal CF":"Villarreal",
    "Sevilla FC":"Sevilla","Real Betis Balompié":"Betis",
    "Inter Milan":"Inter","AC Milan":"Milan","Juventus FC":"Juventus",
    "SSC Napoli":"Napoli","AS Roma":"Roma","SS Lazio":"Lazio",
    "ACF Fiorentina":"Fiorentina","Atalanta BC":"Atalanta",
    "Paris Saint-Germain FC":"Paris SG","Olympique de Marseille":"Marseille",
    "Olympique Lyonnais":"Lyon","AS Monaco FC":"Monaco","LOSC Lille":"Lille",
}


def fetch_todays_fixtures(api_key: str | None = None, league_key: str | None = None) -> tuple[list[dict], list[str]]:
    """
    Fetch today's fixtures from football-data.org API.

    Args:
        api_key:    football-data.org API key (free tier)
        league_key: Filter to one league (None = all)

    Returns:
        (fixtures list, errors list)
    """
    import requests, time, os

    if not api_key:
        api_key = os.environ.get("FOOTBALL_DATA_KEY", "").strip()

    today    = datetime.date.today().isoformat()
    leagues  = ({league_key: FD_ORG_COMP[league_key]}
                if league_key and league_key in FD_ORG_COMP
                else FD_ORG_COMP)
    headers  = {"X-Auth-Token": api_key} if api_key else {}
    fixtures = []
    errors   = []

    for lk, code in leagues.items():
        try:
            url  = (f"https://api.football-data.org/v4/competitions/{code}/matches"
                    f"?dateFrom={today}&dateTo={today}")
            resp = requests.get(url, headers=headers, timeout=12)
            time.sleep(0.7)

            if resp.status_code == 200:
                for m in resp.json().get("matches", []):
                    ht_raw = m["homeTeam"]["name"]
                    at_raw = m["awayTeam"]["name"]
                    fixtures.append({
                        "home_team":  FD_ORG_TO_PARQUET.get(ht_raw, ht_raw),
                        "away_team":  FD_ORG_TO_PARQUET.get(at_raw, at_raw),
                        "home_raw":   ht_raw,
                        "away_raw":   at_raw,
                        "league_key": lk,
                        "kickoff":    m.get("utcDate", "")[:16],
                        "status":     m.get("status", "SCHEDULED"),
                        "matchday":   m.get("matchday", ""),
                    })
            elif resp.status_code == 403:
                errors.append(f"{lk}: Invalid API key")
            elif resp.status_code == 429:
                errors.append(f"{lk}: Rate limited — wait 60s and refresh")
            else:
                errors.append(f"{lk}: HTTP {resp.status_code}")

        except Exception as e:
            errors.append(f"{lk}: {e}")

    return fixtures, errors


def predict_today(api_key: str | None = None, league_key: str | None = None) -> list[dict]:
    """
    Fetch today's fixtures and run match_result predictions on all of them.
    Sorts output HIGH → MEDIUM → LOW confidence.
    """
    today        = datetime.date.today().isoformat()
    fixtures, _  = fetch_todays_fixtures(api_key=api_key, league_key=league_key)

    if not fixtures:
        logger.warning(f"No fixtures for {today}")
        return []

    features_df = load_features()
    results     = []

    for fix in fixtures:
        try:
            pred = predict_market(
                fix["home_team"], fix["away_team"], fix["league_key"],
                market="match_result",
                match_date=today,
                features_df=features_df,
            )
            pred["kickoff"]  = fix.get("kickoff", "")
            pred["status"]   = fix.get("status", "")
            pred["matchday"] = fix.get("matchday", "")
            results.append(pred)
        except Exception as e:
            logger.error(f"✗ {fix['home_team']} vs {fix['away_team']}: {e}")

    conf_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    results.sort(key=lambda x: (
        conf_order.get(x.get("confidence", "LOW"), 2),
        -x.get("top_probability", 0)
    ))
    return results


# ═══════════════════════════════════════════════════════
# SECTION 7 — PRINT HELPERS
# ═══════════════════════════════════════════════════════

def print_prediction(pred: dict, compact: bool = False) -> None:
    if "error" in pred:
        print(f"  ✗ {pred.get('home_team','?')} vs {pred.get('away_team','?')}: {pred['error']}")
        return
    icon = {"HIGH": "🟢", "MEDIUM": "🟡", "LOW": "🔴"}.get(pred["confidence"], "⚪")
    if compact:
        probs = pred["probabilities"]
        vals  = list(probs.values())
        keys  = list(probs.keys())
        print(f"  {icon} {pred['home_team']:<22} vs {pred['away_team']:<22} | "
              f"{pred['league']:<10} | "
              f"{keys[0]}:{vals[0]:.0%} {keys[1]}:{vals[1]:.0%} {keys[2]}:{vals[2]:.0%}"
              f"  → {pred['top_outcome']} ({pred['top_probability']:.0%})")
    else:
        print(f"\n{'='*60}")
        print(f"  {pred['home_team']}  vs  {pred['away_team']}")
        print(f"  {pred['league']}  |  {pred['market']}")
        print(f"{'='*60}")
        for outcome, prob in pred["probabilities"].items():
            print(f"  {outcome:<15} {prob:.1%}  {'█'*int(prob*30)}")
        print(f"\n  TOP PICK:   {pred['top_outcome']}  ({pred['top_probability']:.1%})")
        print(f"  CONFIDENCE: {icon} {CONFIDENCE_MAP.get(pred['confidence'])}")
        if pred.get("shap_values"):
            print("\n  TOP REASONS (SHAP):")
            for i, s in enumerate(pred["shap_values"], 1):
                print(f"    {i}. {s['feature']:<30} val={s['feature_value']:>8.3f}  "
                      f"impact={s['shap_impact']:>+.4f}  {s['direction']}")
        print(f"{'='*60}")


def print_today_summary(predictions: list[dict]) -> None:
    today = datetime.date.today().isoformat()
    print(f"\n{'═'*85}")
    print(f"  TODAY'S PREDICTIONS  —  {today}")
    print(f"{'═'*85}")
    print(f"  {'CONF':<8} {'HOME':<22} {'AWAY':<22} {'LEAGUE':<12} {'TOP PICK':<18} PROB")
    print(f"  {'─'*80}")
    for pred in predictions:
        if "error" in pred:
            continue
        icon = {"HIGH": "🟢 HIGH", "MEDIUM": "🟡 MED", "LOW": "🔴 LOW"}.get(pred["confidence"], "⚪")
        print(f"  {icon:<10} {pred['home_team']:<22} {pred['away_team']:<22} "
              f"{pred['league']:<12} {pred['top_outcome']:<18} {pred['top_probability']:.1%}")
    print(f"{'═'*85}\n")


# ═══════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse, os
    parser = argparse.ArgumentParser(description="Football match predictor")
    parser.add_argument("--home",    default=None)
    parser.add_argument("--away",    default=None)
    parser.add_argument("--league",  default=None)
    parser.add_argument("--market",  default="match_result",
                        choices=["match_result", "btts", "over_under", "corners", "all"])
    parser.add_argument("--date",    default=None)
    parser.add_argument("--today",   action="store_true",
                        help="Predict all matches today using live API")
    args = parser.parse_args()

    PATHS.create_all()
    api_key = os.environ.get("FOOTBALL_DATA_KEY", "").strip()

    if args.today:
        preds = predict_today(api_key=api_key, league_key=args.league)
        if preds:
            print_today_summary(preds)
            print("\n📊 DETAILED — Most confident prediction:")
            print_prediction(preds[0], compact=False)
        else:
            print("No predictions generated for today.")

    elif args.home and args.away and args.league:
        if args.market == "all":
            result = predict_all_markets(
                args.home, args.away, args.league,
                match_date=args.date,
            )
            print(f"\n{'='*60}")
            print(f"  {result['home_team']} vs {result['away_team']} — {result['league']}")
            print(f"  Verdict: {result['model_verdict']}")
            print(f"{'='*60}")
            for mkey, mresult in result["markets"].items():
                print(f"\n  [{mresult['market']}]")
                for outcome, prob in mresult["probabilities"].items():
                    print(f"    {outcome:<15} {prob:.1%}")
        else:
            pred = predict_market(
                args.home, args.away, args.league,
                market=args.market,
                match_date=args.date,
            )
            print_prediction(pred, compact=False)
    else:
        parser.print_help()
        print("\nExamples:")
        print("  python src/predict.py --today")
        print("  python src/predict.py --today --league EPL")
        print("  python src/predict.py --home Arsenal --away Chelsea --league EPL --market all")