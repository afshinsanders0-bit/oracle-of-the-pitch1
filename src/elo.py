"""
elo.py — Dynamic ELO Rating Calculator
========================================
PURPOSE:
    Calculate ELO ratings that update after every match result.
    Attach pre-match ELO ratings to every row in the master DataFrame
    so feature engineering can use them as predictors.

KEY DISCIPLINE — NO DATA LEAKAGE:
    For match N, we record the ELO ratings as they stood BEFORE match N
    was played. The rating only updates AFTER we process the result.
    This mirrors real-world conditions: you never know the outcome
    before you place the bet.

ELO FORMULA:
    Expected score for home team:
        E_home = 1 / (1 + 10^((R_away - R_home - HOME_ADVANTAGE) / 400))

    Rating update after match:
        R_new = R_old + K * (actual_score - expected_score)

    Where actual_score:
        Win  = 1.0
        Draw = 0.5
        Loss = 0.0

INPUTS:
    - Master DataFrame (from data_loader.py), sorted chronologically

OUTPUTS:
    - Same DataFrame with columns added:
        home_elo_before   : Home team ELO rating BEFORE this match
        away_elo_before   : Away team ELO rating BEFORE this match
        elo_diff          : home_elo_before - away_elo_before
        home_elo_after    : Home team ELO rating AFTER this match (for audit)
        away_elo_after    : Away team ELO rating AFTER this match (for audit)
    - Final ratings dict: team → current ELO (for live predictions)

USAGE:
    from src.elo import add_elo_ratings, get_current_ratings
    df = add_elo_ratings(df)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from loguru import logger
from tqdm import tqdm

from config import ELO as ELO_CONFIG, PATHS, LOGGING, SEASONS

# ── Logging ────────────────────────────────────────────────────────────────────
logger.remove()
logger.add(sys.stderr, level=LOGGING.LEVEL, format=LOGGING.FORMAT, colorize=True)


# ═══════════════════════════════════════════════════════
# SECTION 1 — CORE ELO MATH
# ═══════════════════════════════════════════════════════

def expected_score(rating_home: float, rating_away: float) -> float:
    """
    Calculate the expected score (win probability) for the HOME team.

    Home advantage is baked in by adding HOME_ADVANTAGE to the home team's
    effective rating before computing the expectation. This means the home
    team's rating needs to be ~100 points lower to have a 50/50 expectation.

    Args:
        rating_home: Current ELO rating of the home team
        rating_away: Current ELO rating of the away team

    Returns:
        Float in [0, 1] — expected score for home team.
        0.5 = equal chance, 0.7 = home team is clear favourite.

    Example:
        >>> expected_score(1500, 1500)
        0.64  # Home advantage makes home team favourite even at equal ratings
    """
    # The 400 divisor is standard ELO — controls how fast ratings diverge
    exponent = (rating_away - rating_home - ELO_CONFIG.HOME_ADVANTAGE) / 400
    return 1.0 / (1.0 + 10.0 ** exponent)


def update_ratings(
    rating_home: float,
    rating_away: float,
    result: str,
    k_factor: float = ELO_CONFIG.K_FACTOR,
) -> tuple[float, float]:
    """
    Update both teams' ELO ratings after a match result.

    Args:
        rating_home: Home team's ELO before the match
        rating_away: Away team's ELO before the match
        result:      'H' (home win), 'D' (draw), or 'A' (away win)
        k_factor:    How much ratings shift per match (default from config)

    Returns:
        Tuple of (new_home_rating, new_away_rating)

    Raises:
        ValueError: If result is not 'H', 'D', or 'A'
    """
    # Map result to actual scores for ELO formula
    actual_scores = {"H": (1.0, 0.0), "D": (0.5, 0.5), "A": (0.0, 1.0)}

    if result not in actual_scores:
        raise ValueError(f"Invalid result '{result}'. Must be 'H', 'D', or 'A'.")

    actual_home, actual_away = actual_scores[result]

    # Expected score for each team
    exp_home = expected_score(rating_home, rating_away)
    exp_away = 1.0 - exp_home  # Away expected is just the complement

    # Update ratings: K * (actual - expected)
    new_home = rating_home + k_factor * (actual_home - exp_home)
    new_away = rating_away + k_factor * (actual_away - exp_away)

    return round(new_home, 2), round(new_away, 2)


# ═══════════════════════════════════════════════════════
# SECTION 2 — SEASON REGRESSION
# ═══════════════════════════════════════════════════════

def apply_season_regression(
    ratings: dict[str, float],
    league_teams: set[str],
    league_mean: float = ELO_CONFIG.DEFAULT_RATING,
) -> dict[str, float]:
    """
    At the start of each new season, regress all team ratings toward the
    league mean. This prevents ratings from drifting too far over time and
    handles promoted/relegated teams more gracefully.

    Regression formula: R_new = R_old + REGRESSION_FACTOR * (mean - R_old)
    With REGRESSION_FACTOR = 0.5, a team rated 1700 gets pulled to 1600
    (halfway back to 1500 mean).

    For teams new to the league (promoted teams), they start at the league
    mean rather than inheriting a rating from a lower division.

    Args:
        ratings:      Current ratings dict {team_name: elo_rating}
        league_teams: Set of teams participating this season (after promotion/relegation)
        league_mean:  Target mean to regress toward (default: ELO default 1500)

    Returns:
        Updated ratings dict with regressed values.
    """
    updated = {}
    factor  = ELO_CONFIG.PROMOTION_REGRESSION  # 0.5 = halfway to mean

    for team in league_teams:
        if team in ratings:
            # Existing team: regress toward mean
            old_rating = ratings[team]
            new_rating = old_rating + factor * (league_mean - old_rating)
            updated[team] = round(new_rating, 2)
        else:
            # New team (promoted): start at league mean
            updated[team] = league_mean
            logger.debug(f"New team detected: {team} → starting ELO {league_mean}")

    return updated


# ═══════════════════════════════════════════════════════
# SECTION 3 — MAIN ELO PIPELINE
# ═══════════════════════════════════════════════════════

def add_elo_ratings(
    df: pd.DataFrame,
    per_league: bool = True,
) -> pd.DataFrame:
    """
    Add pre-match ELO ratings to every row in the master DataFrame.

    Processes matches in strict chronological order. For each match:
        1. Record CURRENT ratings as 'before' values (this is the feature)
        2. Update ratings based on match result
        3. Move to next match

    This guarantees zero data leakage — features only reflect information
    available before the match was played.

    Args:
        df:          Master DataFrame sorted by date (from data_loader.py)
        per_league:  If True, maintain separate ELO pools per league.
                     If False, one global pool (not recommended for multi-league).

    Returns:
        DataFrame with new columns:
            home_elo_before, away_elo_before, elo_diff,
            home_elo_after, away_elo_after

    Note:
        The DataFrame MUST be sorted by date before calling this function.
        build_master_df() in data_loader.py ensures this, but we check again
        here to be safe.
    """
    # ── Safety check: ensure chronological order ───────────────────────────
    if not df["date"].is_monotonic_increasing:
        logger.warning("DataFrame is not sorted by date — sorting now.")
        df = df.sort_values("date").reset_index(drop=True)

    # ── Initialise output columns ──────────────────────────────────────────
    df = df.copy()
    df["home_elo_before"] = np.nan
    df["away_elo_before"] = np.nan
    df["elo_diff"]        = np.nan
    df["home_elo_after"]  = np.nan
    df["away_elo_after"]  = np.nan

    if per_league:
        # One ratings dictionary per league — teams don't bleed between leagues
        df = _add_elo_per_league(df)
    else:
        df = _add_elo_global(df)

    logger.success(
        f"ELO ratings added: {df['home_elo_before'].notna().sum():,} matches processed"
    )
    return df


def _add_elo_per_league(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate ELO ratings separately for each league.
    This is the recommended mode — EPL and La Liga teams don't share ratings.
    """
    all_frames = []

    leagues = df["league_key"].unique()
    logger.info(f"Computing ELO ratings for {len(leagues)} leagues separately ...")

    for league_key in leagues:
        league_df = df[df["league_key"] == league_key].copy()
        logger.info(f"  Processing {league_key}: {len(league_df):,} matches ...")

        rated_df = _process_league_elo(league_df, league_key)
        all_frames.append(rated_df)

    # Recombine all leagues, sort chronologically again
    result = pd.concat(all_frames, ignore_index=True)
    result.sort_values("date", inplace=True)
    result.reset_index(drop=True, inplace=True)
    return result


def _process_league_elo(df: pd.DataFrame, league_key: str) -> pd.DataFrame:
    """
    Process ELO ratings for a single league across all its seasons.

    Applies season-start regression at the beginning of each new season.
    Teams maintain their ratings across seasons (with regression) to reflect
    that good teams from last year are probably still good this year.
    """
    # Rating store: {team_name: current_elo_rating}
    ratings: dict[str, float] = {}

    # Track which season we're currently in (to detect season transitions)
    current_season = None

    # Process row by row in chronological order
    for idx in tqdm(df.index, desc=f"ELO {league_key}", leave=False, unit="match"):
        row         = df.loc[idx]
        home_team   = row["home_team"]
        away_team   = row["away_team"]
        result      = row["result"]
        season      = row["season"]

        # ── Season transition: apply regression at start of each new season ──
        if season != current_season:
            if current_season is not None:
                # New season detected — regress existing ratings toward mean.
                # FIX: select each column as a Series and call .tolist() on it
                # directly. Using df.loc[mask, col].tolist() gives Pylance a
                # concrete list[str]; the previous pattern of concatenating two
                # .loc scalar results confused Pylance into a wide scalar union
                # that has no __iter__ and no .tolist().
                season_mask  = df["season"] == season
                home_teams: list[str] = df.loc[season_mask, "home_team"].tolist()
                away_teams: list[str] = df.loc[season_mask, "away_team"].tolist()
                season_teams: set[str] = set(home_teams + away_teams)

                # Compute current league mean (only teams still in this league).
                # FIX: float() narrows np.mean's return type (floating[Any])
                # to plain float, satisfying apply_season_regression's signature.
                current_team_ratings = [
                    ratings[t] for t in season_teams if t in ratings
                ]
                league_mean = float(
                    np.mean(current_team_ratings)
                    if current_team_ratings
                    else ELO_CONFIG.DEFAULT_RATING
                )
                ratings = apply_season_regression(ratings, season_teams, league_mean)
                logger.debug(
                    f"{league_key} season {season}: regression applied, "
                    f"mean ELO = {league_mean:.1f}"
                )
            current_season = season

        # ── Initialise any team we haven't seen yet ────────────────────────
        if home_team not in ratings:
            ratings[home_team] = ELO_CONFIG.DEFAULT_RATING
        if away_team not in ratings:
            ratings[away_team] = ELO_CONFIG.DEFAULT_RATING

        # ── Step 1: Record PRE-MATCH ratings as features ───────────────────
        home_before = ratings[home_team]
        away_before = ratings[away_team]

        df.loc[idx, "home_elo_before"] = home_before
        df.loc[idx, "away_elo_before"] = away_before
        df.loc[idx, "elo_diff"]        = home_before - away_before

        # ── Step 2: Update ratings using match result ──────────────────────
        new_home, new_away = update_ratings(home_before, away_before, result)

        ratings[home_team] = new_home
        ratings[away_team] = new_away

        # ── Step 3: Record POST-MATCH ratings (for audit/debugging only) ───
        df.loc[idx, "home_elo_after"] = new_home
        df.loc[idx, "away_elo_after"] = new_away

    return df


def _add_elo_global(df: pd.DataFrame) -> pd.DataFrame:
    """
    Single global ELO pool across all leagues.
    Not recommended for production, but available for experimentation.
    """
    ratings: dict[str, float] = {}
    logger.info("Computing ELO ratings globally (single pool for all leagues) ...")

    for idx in tqdm(df.index, desc="ELO (global)", unit="match"):
        row       = df.loc[idx]
        home_team = row["home_team"]
        away_team = row["away_team"]
        result    = row["result"]

        if home_team not in ratings:
            ratings[home_team] = ELO_CONFIG.DEFAULT_RATING
        if away_team not in ratings:
            ratings[away_team] = ELO_CONFIG.DEFAULT_RATING

        home_before = ratings[home_team]
        away_before = ratings[away_team]

        df.loc[idx, "home_elo_before"] = home_before
        df.loc[idx, "away_elo_before"] = away_before
        df.loc[idx, "elo_diff"]        = home_before - away_before

        new_home, new_away = update_ratings(home_before, away_before, result)
        ratings[home_team] = new_home
        ratings[away_team] = new_away

        df.loc[idx, "home_elo_after"] = new_home
        df.loc[idx, "away_elo_after"] = new_away

    return df


# ═══════════════════════════════════════════════════════
# SECTION 4 — LIVE RATING QUERIES
# ═══════════════════════════════════════════════════════

def get_current_ratings(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    """
    Extract the most recent ELO rating for every team in every league.
    Used by the live predictor to rate upcoming matches.

    Args:
        df: DataFrame with elo columns already added by add_elo_ratings()

    Returns:
        Nested dict: {league_key: {team_name: current_elo}}

    Example:
        ratings = get_current_ratings(df)
        ratings['EPL']['Arsenal']   # → 1623.5
        ratings['LA_LIGA']['Barcelona']  # → 1711.2
    """
    if "home_elo_after" not in df.columns:
        raise ValueError(
            "ELO columns not found. Run add_elo_ratings(df) first."
        )

    current_ratings: dict[str, dict[str, float]] = {}

    for league_key in df["league_key"].unique():
        league_df = df[df["league_key"] == league_key]
        league_ratings: dict[str, float] = {}

        # For each team, take their most recent post-match rating
        # Using home_elo_after and away_elo_after — whichever is more recent
        all_teams = set(league_df["home_team"].unique()) | set(league_df["away_team"].unique())

        for team in all_teams:
            # Get all matches involving this team
            home_mask  = league_df["home_team"] == team
            away_mask  = league_df["away_team"] == team

            home_rows  = league_df[home_mask][["date", "home_elo_after"]].rename(
                columns={"home_elo_after": "elo_after"}
            )
            away_rows  = league_df[away_mask][["date", "away_elo_after"]].rename(
                columns={"away_elo_after": "elo_after"}
            )

            all_rows   = pd.concat([home_rows, away_rows]).sort_values("date")

            if not all_rows.empty and all_rows["elo_after"].notna().any():
                latest_elo = all_rows["elo_after"].dropna().iloc[-1]
                league_ratings[team] = float(latest_elo)

        current_ratings[league_key] = league_ratings

    return current_ratings


def get_team_elo_history(
    df: pd.DataFrame,
    team: str,
    league_key: str,
) -> pd.DataFrame:
    """
    Get the full ELO rating history for a specific team.
    Useful for plotting how a team's strength has evolved over time.

    Args:
        df:          Full master DataFrame with ELO columns
        team:        Team name (must match exactly as stored in data)
        league_key:  League key (e.g. 'EPL')

    Returns:
        DataFrame with columns: date, opponent, result, elo_before, elo_after
    """
    league_df = df[df["league_key"] == league_key].copy()

    # Matches where team played at home
    home_rows = league_df[league_df["home_team"] == team][[
        "date", "away_team", "result", "home_elo_before", "home_elo_after"
    ]].rename(columns={
        "away_team": "opponent",
        "home_elo_before": "elo_before",
        "home_elo_after": "elo_after",
    })
    home_rows["venue"] = "Home"

    # Matches where team played away
    away_rows = league_df[league_df["away_team"] == team][[
        "date", "home_team", "result", "away_elo_before", "away_elo_after"
    ]].rename(columns={
        "home_team": "opponent",
        "away_elo_before": "elo_before",
        "away_elo_after": "elo_after",
    })
    away_rows["venue"] = "Away"

    history = pd.concat([home_rows, away_rows]).sort_values("date").reset_index(drop=True)

    if history.empty:
        logger.warning(f"No ELO history found for {team} in {league_key}")

    return history


# ═══════════════════════════════════════════════════════
# SECTION 5 — ELO LEADERBOARD
# ═══════════════════════════════════════════════════════

def elo_leaderboard(df: pd.DataFrame, league_key: str | None = None) -> pd.DataFrame:
    """
    Show current ELO rankings for all teams, sorted highest to lowest.

    Args:
        df:          Master DataFrame with ELO columns
        league_key:  Filter to one league (optional)

    Returns:
        DataFrame: rank, team, league, current_elo, matches_played
    """
    current = get_current_ratings(df)
    rows    = []

    for league, teams in current.items():
        if league_key and league != league_key:
            continue

        for team, elo in teams.items():
            # Count matches played
            team_mask    = (df["home_team"] == team) | (df["away_team"] == team)
            league_mask  = df["league_key"] == league
            matches      = (team_mask & league_mask).sum()

            rows.append({
                "league":          league,
                "team":            team,
                "current_elo":     round(elo, 1),
                "matches_played":  int(matches),
            })

    if not rows:
        return pd.DataFrame()

    board = (
        pd.DataFrame(rows)
        .sort_values("current_elo", ascending=False)
        .reset_index(drop=True)
    )
    board.index += 1  # Start ranking at 1
    board.index.name = "rank"

    return board


# ═══════════════════════════════════════════════════════
# SECTION 6 — SAVE & LOAD
# ═══════════════════════════════════════════════════════

def save_elo_df(df: pd.DataFrame) -> None:
    """Save the ELO-enriched DataFrame to parquet."""
    path = PATHS.PROCESSED / "master_with_elo.parquet"
    # FIX: Pylance stubs missing 'pyarrow' — suppress false positive
    df.to_parquet(path, index=False, engine="pyarrow")  # type: ignore[call-overload]
    logger.success(f"Saved ELO-enriched DataFrame: {path}")


def load_elo_df() -> pd.DataFrame:
    """Load the ELO-enriched DataFrame from parquet."""
    path = PATHS.PROCESSED / "master_with_elo.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"ELO parquet not found: {path}\n"
            "Run: python src/elo.py    to build it first."
        )
    # FIX: Pylance stubs missing 'pyarrow' — suppress false positive
    return pd.read_parquet(path, engine="pyarrow")  # type: ignore[call-overload]


# ═══════════════════════════════════════════════════════
# MAIN — run as script to attach ELO to master parquet
# ═══════════════════════════════════════════════════════

if __name__ == "__main__":
    from data_loader import load_master_df

    PATHS.create_all()

    logger.info("Loading master DataFrame ...")
    master = load_master_df()

    logger.info("Adding ELO ratings ...")
    master_elo = add_elo_ratings(master, per_league=True)

    # Print leaderboard snapshot
    logger.info("Current ELO leaderboard (EPL):")
    board = elo_leaderboard(master_elo, league_key="EPL")
    print(board.head(10).to_string())

    save_elo_df(master_elo)
    logger.success("Done. ELO ratings saved to data/processed/master_with_elo.parquet")