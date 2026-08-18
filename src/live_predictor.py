"""
live_predictor.py — Real-Time In-Play Probability Updates
===========================================================
Scrapes live match stats from Sofascore every 5 minutes during matches.
Updates win/draw/loss probabilities as the match progresses.
Triggers VALUE ALERT when live odds drift vs our live model probability.
Stores live match data for post-match model retraining.

Run:  python src/live_predictor.py               ← monitor all live matches today
      python src/live_predictor.py --match-id 12345678  ← monitor one specific match
"""

import sys
import json
import time
import datetime
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import requests
import pandas as pd
import numpy as np
from loguru import logger

from config import PATHS, SCRAPING, LOGGING, LEAGUES, VALUE_BET

logger.remove()
logger.add(sys.stderr, level=LOGGING.LEVEL, format=LOGGING.FORMAT, colorize=True)
logger.add(PATHS.LOG_FILE, level="DEBUG", format=LOGGING.FORMAT,
           rotation=LOGGING.ROTATION, retention=LOGGING.RETENTION)

# ── Sofascore API (unofficial — reverse engineered public endpoints) ───────────
SOFASCORE_API = "https://api.sofascore.com/api/v1"
SOFASCORE_HEADERS = {
    **SCRAPING.DEFAULT_HEADERS,
    "Referer": "https://www.sofascore.com/",
    "Accept":  "application/json",
    "Cache-Control": "no-cache",
}


# ═══════════════════════════════════════════════════════
# SECTION 1 — SOFASCORE API HELPERS
# ═══════════════════════════════════════════════════════

def _api_get(endpoint: str, retries: int = 3) -> dict | None:
    """
    Make a GET request to the Sofascore API.

    Args:
        endpoint: API path e.g. '/sport/football/scheduled-events/2024-05-01'

    Returns:
        Parsed JSON dict, or None if request failed.
    """
    url = f"{SOFASCORE_API}{endpoint}"
    for attempt in range(retries):
        try:
            time.sleep(1.5)   # Sofascore rate limit — be polite
            resp = requests.get(url, headers=SOFASCORE_HEADERS,
                                timeout=SCRAPING.REQUEST_TIMEOUT)
            if resp.status_code == 200:
                return resp.json()
            elif resp.status_code == 429:
                logger.warning("Rate limited by Sofascore — sleeping 30s")
                time.sleep(30)
            else:
                logger.warning(f"Sofascore HTTP {resp.status_code}: {url}")
        except requests.RequestException as e:
            logger.warning(f"Sofascore request error: {e} (attempt {attempt+1})")
            time.sleep(5)
    return None


def get_live_matches(league_keys: list[str] | None = None) -> list[dict]:
    """
    Fetch all currently live football matches from Sofascore.

    Args:
        league_keys: Filter to specific leagues (e.g. ['EPL', 'LA_LIGA'])
                     If None, returns all live football matches.

    Returns:
        List of match dicts with keys:
            match_id, home_team, away_team, league, minute,
            home_score, away_score, status
    """
    data = _api_get("/sport/football/events/live")
    if not data or "events" not in data:
        logger.warning("No live matches data from Sofascore.")
        return []

    # Sofascore tournament IDs for our leagues
    tournament_ids = {
        "EPL":           17,    # Premier League
        "ENG_CHAMP":     18,    # Championship
        "LA_LIGA":       8,     # La Liga
        "SERIE_A":       23,    # Serie A
        "LIGUE_1":       34,    # Ligue 1
        "BUNDESLIGA":    3,     # Bundesliga
    }
    target_ids = set()
    if league_keys:
        for k in league_keys:
            tid = tournament_ids.get(k)
            if tid: target_ids.add(tid)

    matches = []
    for event in data.get("events", []):
        try:
            t_id = event.get("tournament", {}).get("uniqueTournament", {}).get("id")
            if target_ids and t_id not in target_ids:
                continue

            status    = event.get("status", {}).get("description", "")
            score     = event.get("homeScore", {}), event.get("awayScore", {})

            matches.append({
                "match_id":    event.get("id"),
                "home_team":   event.get("homeTeam", {}).get("name", ""),
                "away_team":   event.get("awayTeam", {}).get("name", ""),
                "league":      event.get("tournament", {}).get("name", ""),
                "tournament_id": t_id,
                "minute":      event.get("time", {}).get("played", 0),
                "home_score":  score[0].get("current", 0),
                "away_score":  score[1].get("current", 0),
                "status":      status,
            })
        except (KeyError, TypeError):
            continue

    logger.info(f"Live matches found: {len(matches)}")
    return matches


def get_match_stats(match_id: int) -> dict:
    """
    Fetch detailed live stats for a specific match from Sofascore.

    Returns:
        Dict with: possession, shots, shots_on_target, corners,
                   yellow_cards, red_cards, xg (if available),
                   minute, home_score, away_score
    """
    data = _api_get(f"/event/{match_id}/statistics")
    if not data or "statistics" not in data:
        return {}

    stats = {
        "home_possession": 50.0, "away_possession": 50.0,
        "home_shots": 0, "away_shots": 0,
        "home_shots_ot": 0, "away_shots_ot": 0,
        "home_corners": 0, "away_corners": 0,
        "home_yellows": 0, "away_yellows": 0,
        "home_reds": 0, "away_reds": 0,
        "home_xg": 0.0, "away_xg": 0.0,
    }

    # Sofascore nests stats in periods — aggregate all periods
    stat_map = {
        "Ball possession":        ("home_possession", "away_possession"),
        "Total shots":            ("home_shots", "away_shots"),
        "Shots on target":        ("home_shots_ot", "away_shots_ot"),
        "Corner kicks":           ("home_corners", "away_corners"),
        "Yellow cards":           ("home_yellows", "away_yellows"),
        "Red cards":              ("home_reds", "away_reds"),
        "Expected goals":         ("home_xg", "away_xg"),
    }

    for period in data.get("statistics", []):
        for group in period.get("groups", []):
            for item in group.get("statisticsItems", []):
                name = item.get("name", "")
                if name in stat_map:
                    home_key, away_key = stat_map[name]
                    try:
                        h_val = item.get("home", "0").replace("%","")
                        a_val = item.get("away", "0").replace("%","")
                        stats[home_key] = float(h_val or 0)
                        stats[away_key] = float(a_val or 0)
                    except (ValueError, AttributeError):
                        pass

    return stats


def get_match_odds(match_id: int) -> dict:
    """
    Fetch current bookmaker odds for a live match from Sofascore.

    Returns:
        Dict with: odds_home, odds_draw, odds_away (decimal)
    """
    data = _api_get(f"/event/{match_id}/odds/1/all")  # 1 = 1X2 market
    if not data:
        return {}

    try:
        choices = data.get("markets", [{}])[0].get("choices", [])
        odds    = {}
        for choice in choices:
            name = choice.get("name", "").lower()
            frac = choice.get("fractionalValue", "")
            # Convert fractional to decimal: 'n/d' → n/d + 1
            if "/" in str(frac):
                n, d = frac.split("/")
                decimal = round(int(n) / int(d) + 1, 2)
            else:
                decimal = float(frac or 0) if frac else 0.0

            if "home" in name or name == "1":
                odds["odds_home"] = decimal
            elif "draw" in name or name == "x":
                odds["odds_draw"] = decimal
            elif "away" in name or name == "2":
                odds["odds_away"] = decimal
        return odds
    except (IndexError, KeyError, ValueError, ZeroDivisionError):
        return {}


# ═══════════════════════════════════════════════════════
# SECTION 2 — IN-PLAY PROBABILITY MODEL
# ═══════════════════════════════════════════════════════

def compute_live_probabilities(
    pre_match_proba: np.ndarray,
    minute: int,
    home_score: int,
    away_score: int,
    home_xg: float = 0.0,
    away_xg: float = 0.0,
    home_shots_ot: int = 0,
    away_shots_ot: int = 0,
) -> dict:
    """
    Update win/draw/loss probabilities based on current match state.

    Method:
        Uses a weighted blend of:
        1. Pre-match model probability (weight decreases as match progresses)
        2. In-play Poisson estimate based on current score + xG pace

        As the match approaches 90 minutes, the in-play evidence dominates.
        At minute 0, the pre-match model dominates.

    Args:
        pre_match_proba: np.array [P(home), P(draw), P(away)] from model
        minute:          Current match minute (0-90+)
        home_score:      Current home goals
        away_score:      Current away goals
        home_xg:         Cumulative home xG so far
        away_xg:         Cumulative away xG so far
        home_shots_ot:   Home shots on target
        away_shots_ot:   Away shots on target

    Returns:
        Dict with: p_home, p_draw, p_away, method, minute
    """
    minute      = max(1, min(minute, 90))
    remaining   = max(90 - minute, 1)

    # Weight of pre-match model decreases linearly as match progresses
    # At minute 0: 100% pre-match; at minute 90: 10% pre-match
    pre_weight  = max(0.10, 1.0 - (minute / 90) * 0.90)
    live_weight = 1.0 - pre_weight

    # ── In-play Poisson estimate ───────────────────────────────────────────
    # Estimate remaining goals using current xG pace extrapolated to 90 min
    # Use xG if available, else fall back to shots on target proxy
    if home_xg > 0 or away_xg > 0:
        home_rate_per_min = home_xg / minute
        away_rate_per_min = away_xg / minute
    else:
        # Rough proxy: each shot on target ≈ 0.3 xG
        home_rate_per_min = (home_shots_ot * 0.3) / minute
        away_rate_per_min = (away_shots_ot * 0.3) / minute

    # Expected remaining goals
    lambda_home = home_rate_per_min * remaining
    lambda_away = away_rate_per_min * remaining

    # Simulate outcome probabilities using Poisson
    max_goals   = 8
    p_home_live = 0.0
    p_draw_live = 0.0
    p_away_live = 0.0

    from scipy.stats import poisson
    for h in range(max_goals):
        for a in range(max_goals):
            prob = poisson.pmf(h, lambda_home) * poisson.pmf(a, lambda_away)
            final_h = home_score + h
            final_a = away_score + a
            if final_h > final_a:
                p_home_live += prob
            elif final_h == final_a:
                p_draw_live += prob
            else:
                p_away_live += prob

    live_proba = np.array([p_home_live, p_draw_live, p_away_live])
    # Normalise (Poisson truncation may leave sum slightly off 1)
    live_proba = live_proba / live_proba.sum()

    # ── Blend pre-match and live ───────────────────────────────────────────
    blended = pre_weight * pre_match_proba + live_weight * live_proba
    blended = blended / blended.sum()   # ensure sum = 1

    return {
        "p_home":      round(float(blended[0]), 4),
        "p_draw":      round(float(blended[1]), 4),
        "p_away":      round(float(blended[2]), 4),
        "pre_weight":  round(pre_weight, 3),
        "live_weight": round(live_weight, 3),
        "minute":      minute,
        "score":       f"{home_score}-{away_score}",
    }


# ═══════════════════════════════════════════════════════
# SECTION 3 — VALUE ALERT ENGINE
# ═══════════════════════════════════════════════════════

def check_live_value_alert(
    live_proba: dict,
    live_odds: dict,
    min_edge: float = VALUE_BET.MIN_EDGE,
) -> list[dict]:
    """
    Compare our live model probability with current bookmaker odds.
    Trigger a VALUE ALERT when our probability exceeds fair odds by min_edge.

    Args:
        live_proba: Dict from compute_live_probabilities()
        live_odds:  Dict from get_match_odds() {odds_home, odds_draw, odds_away}
        min_edge:   Minimum edge to trigger alert (default 5%)

    Returns:
        List of alert dicts (empty if no value found).
    """
    alerts = []
    mapping = [
        ("Home Win", "p_home", "odds_home"),
        ("Draw",     "p_draw", "odds_draw"),
        ("Away Win", "p_away", "odds_away"),
    ]

    for outcome, prob_key, odds_key in mapping:
        model_prob = live_proba.get(prob_key, 0)
        odds       = live_odds.get(odds_key, 0)

        if odds <= 1.0:
            continue

        # Remove overround: fair_prob = 1/odds (simplified — single market)
        fair_prob = 1.0 / odds
        edge      = model_prob - fair_prob
        ev        = model_prob * odds - 1.0

        if edge >= min_edge and model_prob >= VALUE_BET.MIN_MODEL_PROB:
            alerts.append({
                "outcome":    outcome,
                "model_prob": round(model_prob, 4),
                "fair_prob":  round(fair_prob, 4),
                "odds":       odds,
                "edge":       round(edge, 4),
                "ev":         round(ev, 4),
                "minute":     live_proba.get("minute"),
                "score":      live_proba.get("score"),
            })

    return alerts


# ═══════════════════════════════════════════════════════
# SECTION 4 — LIVE MATCH STORAGE
# ═══════════════════════════════════════════════════════

def save_live_snapshot(match_id: int, snapshot: dict) -> None:
    """
    Save one scrape's worth of live data to the live cache folder.
    Used for post-match model retraining and analysis.

    Args:
        match_id: Sofascore match ID
        snapshot: Dict with all live stats, probabilities, odds, alerts
    """
    PATHS.LIVE.mkdir(parents=True, exist_ok=True)
    path = PATHS.LIVE / f"match_{match_id}.jsonl"

    snapshot["scraped_at"] = datetime.datetime.utcnow().isoformat()

    with open(path, "a") as f:
        f.write(json.dumps(snapshot) + "\n")


def load_live_history(match_id: int) -> list[dict]:
    """Load all stored snapshots for a match (for timeline analysis)."""
    path = PATHS.LIVE / f"match_{match_id}.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().strip().splitlines()]


# ═══════════════════════════════════════════════════════
# SECTION 5 — MAIN MONITOR LOOP
# ═══════════════════════════════════════════════════════

def monitor_match(
    match_id: int,
    pre_match_proba: np.ndarray,
    interval_seconds: int = SCRAPING.LIVE_SCRAPE_INTERVAL,
    max_minutes: int = 110,
) -> None:
    """
    Monitor one live match continuously, updating probabilities every interval.

    Args:
        match_id:         Sofascore match ID
        pre_match_proba:  [P(home), P(draw), P(away)] from our trained model
        interval_seconds: How often to scrape (default 300s = 5 minutes)
        max_minutes:      Stop monitoring after this many minutes
    """
    logger.info(f"Starting live monitor for match {match_id} "
                f"(interval={interval_seconds}s)")
    start_time = time.time()

    while True:
        elapsed_min = (time.time() - start_time) / 60
        if elapsed_min > max_minutes:
            logger.info(f"Match {match_id}: max_minutes={max_minutes} reached. Stopping.")
            break

        # ── Scrape current stats ───────────────────────────────────────────
        stats = get_match_stats(match_id)
        odds  = get_match_odds(match_id)

        if not stats:
            logger.warning(f"No stats returned for match {match_id} — will retry.")
            time.sleep(interval_seconds)
            continue

        minute     = int(stats.get("minute", elapsed_min))
        home_score = int(stats.get("home_score", 0))
        away_score = int(stats.get("away_score", 0))

        # ── Compute live probabilities ────────────────────────────────────
        live_proba = compute_live_probabilities(
            pre_match_proba = pre_match_proba,
            minute          = minute,
            home_score      = home_score,
            away_score      = away_score,
            home_xg         = float(stats.get("home_xg", 0)),
            away_xg         = float(stats.get("away_xg", 0)),
            home_shots_ot   = int(stats.get("home_shots_ot", 0)),
            away_shots_ot   = int(stats.get("away_shots_ot", 0)),
        )

        # ── Check for value alerts ─────────────────────────────────────────
        alerts = check_live_value_alert(live_proba, odds)

        # ── Print live dashboard ───────────────────────────────────────────
        _print_live_dashboard(match_id, live_proba, stats, odds, alerts)

        # ── Save snapshot to disk ──────────────────────────────────────────
        snapshot = {
            "match_id":    match_id,
            "minute":      minute,
            "score":       f"{home_score}-{away_score}",
            "stats":       stats,
            "odds":        odds,
            "live_proba":  live_proba,
            "alerts":      alerts,
        }
        save_live_snapshot(match_id, snapshot)

        # ── Check if match is over ─────────────────────────────────────────
        if minute >= 90:
            logger.info(f"Match {match_id}: 90+ minutes reached. Monitoring complete.")
            break

        time.sleep(interval_seconds)


def monitor_all_live(league_keys: list[str] | None = None,
                     interval_seconds: int = SCRAPING.LIVE_SCRAPE_INTERVAL) -> None:
    """
    Monitor ALL currently live matches in specified leagues.
    Runs in a single-threaded loop — refreshes every interval.

    For production, each match should run in a separate thread/process.
    """
    logger.info("Starting live match monitor for all matches ...")

    while True:
        live_matches = get_live_matches(league_keys=league_keys)

        if not live_matches:
            logger.info("No live matches right now. Checking again in 60s ...")
            time.sleep(60)
            continue

        for match in live_matches:
            mid   = match["match_id"]
            stats = get_match_stats(mid)
            odds  = get_match_odds(mid)

            # Use equal priors if no model loaded (0.45/0.25/0.30 typical base rates)
            pre_proba = np.array([0.45, 0.25, 0.30])

            live_proba = compute_live_probabilities(
                pre_match_proba = pre_proba,
                minute          = match.get("minute", 1),
                home_score      = match.get("home_score", 0),
                away_score      = match.get("away_score", 0),
                home_xg         = float(stats.get("home_xg", 0)),
                away_xg         = float(stats.get("away_xg", 0)),
            )

            alerts = check_live_value_alert(live_proba, odds)

            if alerts:
                for alert in alerts:
                    logger.warning(
                        f"🚨 VALUE ALERT | {match['home_team']} vs {match['away_team']} "
                        f"| {alert['minute']}' {alert['score']} "
                        f"| {alert['outcome']} @ {alert['odds']} "
                        f"| model={alert['model_prob']:.1%} edge={alert['edge']:+.1%} EV={alert['ev']:+.3f}"
                    )

            save_live_snapshot(mid, {
                "match": match, "stats": stats, "odds": odds,
                "live_proba": live_proba, "alerts": alerts,
            })

        logger.info(f"Cycle complete. Next update in {interval_seconds}s ...")
        time.sleep(interval_seconds)


def _print_live_dashboard(match_id, live_proba, stats, odds, alerts):
    """Print a live match dashboard to the terminal."""
    score = live_proba.get("score", "?-?")
    min_  = live_proba.get("minute", "?")

    print(f"\n{'─'*55}")
    print(f"  LIVE  Match {match_id}  |  {min_}'  Score: {score}")
    print(f"{'─'*55}")
    print(f"  Home Win : {live_proba['p_home']:.1%}  "
          f"Draw: {live_proba['p_draw']:.1%}  "
          f"Away Win: {live_proba['p_away']:.1%}")
    print(f"  Pre-match weight: {live_proba['pre_weight']:.0%}  "
          f"Live weight: {live_proba['live_weight']:.0%}")

    if stats:
        print(f"  Shots OT: {stats.get('home_shots_ot',0)}-{stats.get('away_shots_ot',0)}  "
              f"Corners: {stats.get('home_corners',0)}-{stats.get('away_corners',0)}  "
              f"xG: {stats.get('home_xg',0):.2f}-{stats.get('away_xg',0):.2f}")

    if odds:
        print(f"  Live odds: H={odds.get('odds_home','?')}  "
              f"D={odds.get('odds_draw','?')}  "
              f"A={odds.get('odds_away','?')}")

    if alerts:
        for a in alerts:
            print(f"\n  🚨 VALUE ALERT: {a['outcome']} @ {a['odds']} "
                  f"| model={a['model_prob']:.1%} edge={a['edge']:+.1%} EV={a['ev']:+.3f}")
    print(f"{'─'*55}")


# ═══════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Live match probability monitor")
    parser.add_argument("--match-id",  type=int, default=None,
                        help="Monitor a specific Sofascore match ID")
    parser.add_argument("--leagues",   nargs="+", default=None,
                        help="Filter leagues e.g. --leagues EPL LA_LIGA")
    parser.add_argument("--interval",  type=int, default=SCRAPING.LIVE_SCRAPE_INTERVAL,
                        help=f"Scrape interval in seconds (default {SCRAPING.LIVE_SCRAPE_INTERVAL})")
    parser.add_argument("--home-prob", type=float, default=0.45,
                        help="Pre-match home win probability from model")
    parser.add_argument("--draw-prob", type=float, default=0.25,
                        help="Pre-match draw probability from model")
    parser.add_argument("--away-prob", type=float, default=0.30,
                        help="Pre-match away win probability from model")
    args = parser.parse_args()

    PATHS.create_all()

    pre_proba = np.array([args.home_prob, args.draw_prob, args.away_prob])
    pre_proba = pre_proba / pre_proba.sum()  # normalise

    if args.match_id:
        monitor_match(
            match_id        = args.match_id,
            pre_match_proba = pre_proba,
            interval_seconds= args.interval,
        )
    else:
        monitor_all_live(
            league_keys      = args.leagues,
            interval_seconds = args.interval,
        )