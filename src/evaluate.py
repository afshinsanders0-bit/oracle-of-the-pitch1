"""
evaluate.py — Model Evaluation, Backtesting & ROI Tracking
FIX: load_features() reads from disk. Does NOT call build_features().
FIX: Loads match_result_current_features.pkl (current-only model).
     Falls back to match_result_features.pkl for the legacy full model.
Run:  python src/evaluate.py
"""
import sys, re
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import joblib
from loguru import logger
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                             brier_score_loss, confusion_matrix)
from sklearn.preprocessing import label_binarize
from config import PATHS, LOGGING, SEASONS, KELLY, VALUE_BET

logger.remove()
logger.add(sys.stderr, level=LOGGING.LEVEL, format=LOGGING.FORMAT, colorize=True)


def sanitise_columns(df):
    seen, new_cols = {}, {}
    for col in df.columns:
        clean = re.sub(r'[^a-zA-Z0-9_]', '_', str(col))
        if clean in seen:
            seen[clean] += 1; clean = f"{clean}_{seen[clean]}"
        else: seen[clean] = 0
        new_cols[col] = clean
    return df.rename(columns=new_cols)


def load_features():
    """Read features parquet from disk. DOES NOT call build_features()."""
    for fname in ["features_master.parquet", "features.parquet",
                  "master_with_elo.parquet", "master.parquet"]:
        p = PATHS.PROCESSED / fname
        if p.exists():
            logger.info(f"Loading: {fname}")
            return pd.read_parquet(p, engine="pyarrow")
    raise FileNotFoundError(
        "No features parquet found.\nRun: python src/update_data.py"
    )


def load_model_and_features():
    """
    Load the match result model and its feature list.

    Priority order for feature list:
      1. match_result_current_features.pkl  — saved by trainCurrentOnly.py
      2. match_result_features.pkl          — saved by legacy train.py

    Raises SystemExit if neither exists.
    """
    model_path = PATHS.MODEL_MATCH_RESULT
    if not model_path.exists():
        logger.error(f"Model not found: {model_path}. Run: python src/trainCurrentOnly.py")
        sys.exit(1)

    model = joblib.load(model_path)

    # Try current-only feature list first, then legacy
    candidates = [
        PATHS.MODELS / "match_result_current_features.pkl",
        PATHS.MODELS / "match_result_features.pkl",
    ]
    for p in candidates:
        if p.exists():
            feature_names = joblib.load(p)
            logger.info(f"Feature list loaded: {p.name}  ({len(feature_names)} features)")
            return model, feature_names

    logger.error(
        "No feature list found. Expected one of:\n"
        + "\n".join(f"  {p}" for p in candidates)
        + "\nRe-run: python src/trainCurrentOnly.py"
    )
    sys.exit(1)


def align_features(df: pd.DataFrame, feature_names: list) -> pd.DataFrame:
    """
    Select and order columns to exactly match the model's feature list.
    Missing columns are filled with -999 (consistent with training imputation).
    Extra columns are dropped silently.
    """
    for col in feature_names:
        if col not in df.columns:
            logger.warning(f"Column '{col}' not in DataFrame — filling with -999")
            df[col] = -999.0
    return df[feature_names].astype(np.float32).fillna(-999)


def ranked_probability_score(y_true_onehot, y_pred_proba):
    """RPS — primary football prediction metric. Lower = better."""
    rps = []
    for t, p in zip(y_true_onehot, y_pred_proba):
        rps.append(np.mean((np.cumsum(p[:-1]) - np.cumsum(t[:-1])) ** 2))
    return float(np.mean(rps))


def evaluate_model(model, X, y_true, label_names=["Home Win", "Draw", "Away Win"]):
    """Compute full evaluation metrics suite."""
    y_pred   = model.predict(X)
    y_proba  = model.predict_proba(X)
    classes  = sorted(np.unique(y_true))
    y_onehot = label_binarize(y_true, classes=classes)
    n_cls    = len(classes)

    m = {}
    m["accuracy"]    = round(accuracy_score(y_true, y_pred), 4)
    m["f1_weighted"] = round(f1_score(y_true, y_pred, average="weighted", zero_division=0), 4)
    f1_per = f1_score(y_true, y_pred, average=None, zero_division=0)
    for i, name in enumerate(label_names[:len(f1_per)]):
        m[f"f1_{name.lower().replace(' ', '_')}"] = round(float(f1_per[i]), 4)
    m["rps"] = round(ranked_probability_score(y_onehot, y_proba), 4)

    try:
        if n_cls == 2:
            m["roc_auc"] = round(roc_auc_score(y_true, y_proba[:, 1]), 4)
        else:
            m["roc_auc_ovr"] = round(roc_auc_score(
                y_onehot, y_proba, multi_class="ovr", average="weighted"), 4)
    except Exception:
        pass

    bs = []
    for i in range(min(n_cls, y_onehot.shape[1])):
        s = brier_score_loss(y_onehot[:, i], y_proba[:, i])
        m[f"brier_{label_names[i].lower().replace(' ', '_')}"] = round(s, 4)
        bs.append(s)
    m["brier_mean"]       = round(float(np.mean(bs)), 4)
    m["confusion_matrix"] = confusion_matrix(y_true, y_pred).tolist()
    return m


def print_metrics(m, title="Model Evaluation"):
    print(f"\n{'='*55}\n  {title}\n{'='*55}")
    print(f"  Accuracy        : {m.get('accuracy', 'N/A')}")
    print(f"  F1 (weighted)   : {m.get('f1_weighted', 'N/A')}")
    print(f"  RPS ▼ (primary) : {m.get('rps', 'N/A')}  (lower = better)")
    print(f"  ROC-AUC         : {m.get('roc_auc_ovr', m.get('roc_auc', 'N/A'))}")
    print(f"  Brier (mean)    : {m.get('brier_mean', 'N/A')}")
    print(f"  Confusion matrix: {m.get('confusion_matrix')}")
    print(f"{'='*55}\n")


def kelly_stake(model_prob, decimal_odds, bankroll, fraction=KELLY.FRACTION):
    b = decimal_odds - 1.0
    if b <= 0: return 0.0
    fk = (b * model_prob - (1 - model_prob)) / b
    if fk <= 0: return 0.0
    return round(min(fraction * fk, KELLY.MAX_STAKE_PCT) * bankroll, 2)


def backtest(features_df, model, feature_names, test_seasons=None,
             starting_bankroll=KELLY.DEFAULT_BANKROLL, staking="kelly",
             flat_stake_pct=0.02):
    """Simulate betting performance on test seasons."""
    test_seasons = test_seasons or SEASONS.TEST
    df = features_df[features_df["season"].isin(test_seasons)].copy()
    df = sanitise_columns(df)

    X = align_features(df, feature_names)

    probas   = model.predict_proba(X)
    bankroll = starting_bankroll
    bet_log  = []
    bh       = [starting_bankroll]

    odds_map = {0: "odds_home", 1: "odds_draw", 2: "odds_away"}
    fair_map = {0: "fair_prob_home", 1: "fair_prob_draw", 2: "fair_prob_away"}

    for i, (idx, row) in enumerate(df.iterrows()):
        proba_row = probas[i]
        top_idx   = int(np.argmax(proba_row))
        top_prob  = float(proba_row[top_idx])
        if top_prob < VALUE_BET.MEDIUM_CONFIDENCE: continue

        oc = odds_map.get(top_idx)
        if not oc or oc not in df.columns or pd.isna(row.get(oc)): continue
        dec_odds = float(row[oc])
        if dec_odds < VALUE_BET.MIN_ODDS or dec_odds > VALUE_BET.MAX_ODDS: continue

        fair_p = float(row.get(fair_map[top_idx], 1 / dec_odds))
        if top_prob - fair_p < VALUE_BET.MIN_EDGE: continue

        ev    = top_prob * dec_odds - 1.0
        stake = kelly_stake(top_prob, dec_odds, bankroll) if staking == "kelly" \
                else round(flat_stake_pct * bankroll, 2)
        if stake < 0.01: continue

        bet_on = {0: "H", 1: "D", 2: "A"}[top_idx]
        won    = (bet_on == row.get("result"))
        profit = stake * (dec_odds - 1) if won else -stake
        bankroll = max(bankroll + profit, 0.01)
        bh.append(bankroll)
        bet_log.append({
            "date": row.get("date"), "league": row.get("league_key"),
            "home_team": row.get("home_team"), "away_team": row.get("away_team"),
            "bet_on": bet_on, "model_prob": round(top_prob, 4),
            "edge": round(top_prob - fair_p, 4), "decimal_odds": dec_odds,
            "ev": round(ev, 4), "stake": stake, "won": won,
            "profit": round(profit, 2), "bankroll": round(bankroll, 2),
        })

    bet_df = pd.DataFrame(bet_log)
    if bet_df.empty:
        logger.warning("No value bets found in backtest.")
        return {"n_bets": 0, "roi": 0.0, "profit_units": 0.0,
                "bankroll_history": bh, "bet_log": bet_df}

    staked  = bet_df["stake"].sum()
    profit  = bet_df["profit"].sum()
    bh_np   = np.array(bh)
    peak    = np.maximum.accumulate(bh_np)
    max_dd  = float(((peak - bh_np) / peak).max())
    profits = bet_df["profit"].values
    sharpe  = (profits.mean() / profits.std()) * np.sqrt(252) if profits.std() > 0 else 0.0

    return {
        "n_bets": len(bet_df), "win_rate": round(bet_df["won"].mean(), 4),
        "roi": round(profit / staked, 4) if staked > 0 else 0.0,
        "profit_units": round(profit, 2), "total_staked": round(staked, 2),
        "final_bankroll": round(bankroll, 2), "max_drawdown": round(max_dd, 4),
        "sharpe_ratio": round(sharpe, 4), "bankroll_history": bh, "bet_log": bet_df,
    }


def print_backtest_summary(r):
    print(f"\n{'='*55}\n  BACKTEST RESULTS\n{'='*55}")
    print(f"  Bets placed    : {r['n_bets']}")
    print(f"  Win rate       : {r.get('win_rate', 0):.1%}")
    print(f"  ROI            : {r.get('roi', 0):+.2%}")
    print(f"  Profit (units) : {r.get('profit_units', 0):+.2f}")
    print(f"  Total staked   : {r.get('total_staked', 0):.2f}")
    print(f"  Final bankroll : {r.get('final_bankroll', 0):.2f}")
    print(f"  Max drawdown   : {r.get('max_drawdown', 0):.1%}")
    print(f"  Sharpe ratio   : {r.get('sharpe_ratio', 0):.3f}")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    PATHS.create_all()

    features_df           = load_features()
    model, feature_names  = load_model_and_features()

    # Build test set — align to model features only
    test_df = features_df[features_df["season"].isin(SEASONS.TEST)].copy()
    test_df = sanitise_columns(test_df)
    X_test  = align_features(test_df, feature_names)

    y_test  = test_df["result"].map({"H": 0, "D": 1, "A": 2}).dropna().astype(int)
    X_test  = X_test.loc[y_test.index]

    logger.info(f"Test set: {len(y_test):,} matches — seasons {SEASONS.TEST}")

    # Metrics
    metrics = evaluate_model(model, X_test, y_test.to_numpy())
    print_metrics(metrics, f"Match Result Model — Test {SEASONS.TEST}")

    # Backtest
    logger.info("Running backtest (Kelly staking) ...")
    results = backtest(features_df, model, feature_names, staking="kelly")
    print_backtest_summary(results)

    if "bet_log" in results and len(results["bet_log"]) > 0:
        log_path = PATHS.DATA / "backtest_bet_log.csv"
        results["bet_log"].to_csv(log_path, index=False)
        logger.success(f"Bet log saved: {log_path}")