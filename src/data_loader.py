"""
data_loader.py — Download, clean, and merge football data for all 5 leagues
=============================================================================
PURPOSE:
    1. Download historical match CSVs from football-data.co.uk
    2. Standardise column names and data types
    3. Derive basic target variables (BTTS, Over 2.5, etc.)
    4. Merge all leagues and seasons into one master DataFrame
    5. Save processed master as parquet for fast downstream loading

INPUTS:
    - Internet connection (first run only — results are cached to data/raw/)
    - config.py (all constants)

OUTPUTS:
    - data/raw/{league_code}_{season}.csv       ← raw downloaded files
    - data/processed/master.parquet             ← cleaned, merged, all leagues
    - data/processed/{league_code}_master.parquet  ← per-league split (convenience)

USAGE:
    python src/data_loader.py                   ← downloads everything, saves parquet
    from src.data_loader import load_master_df  ← load already-processed data
"""

import sys
import time
from pathlib import Path

# Add project root to path so we can import config from anywhere
sys.path.insert(0, str(Path(__file__).parent.parent))

import requests
import pandas as pd
import numpy as np
from loguru import logger
from tqdm import tqdm

from config import (
    LEAGUES, SEASONS, PATHS, URLS,
    FD_COLUMN_MAP, REQUIRED_COLUMNS,
    SCRAPING, LOGGING, RANDOM_SEED
)

# ── Logging setup ──────────────────────────────────────────────────────────────
# Remove default loguru handler and add our configured one
logger.remove()
logger.add(
    sys.stderr,
    level=LOGGING.LEVEL,
    format=LOGGING.FORMAT,
    colorize=True,
)
logger.add(
    PATHS.LOG_FILE,
    level="DEBUG",          # Always log DEBUG to file, INFO to terminal
    format=LOGGING.FORMAT,
    rotation=LOGGING.ROTATION,
    retention=LOGGING.RETENTION,
)


# ═══════════════════════════════════════════════════════
# SECTION 1 — URL BUILDER
# ═══════════════════════════════════════════════════════

def build_fd_url(fd_code: str, season: str) -> str:
    """
    Build the football-data.co.uk download URL for a given league and season.

    Args:
        fd_code:  League code from config.LEAGUES (e.g. 'E0', 'SP1')
        season:   Season string from config.SEASONS (e.g. '2425' for 2024-25)

    Returns:
        Full download URL string.

    Example:
        >>> build_fd_url('E0', '2425')
        'https://www.football-data.co.uk/mmz4281/2425/E0.csv'
    """
    return f"{URLS.BASE_FD}/{season}/{fd_code}.csv"


# ═══════════════════════════════════════════════════════
# SECTION 2 — SINGLE FILE DOWNLOADER
# ═══════════════════════════════════════════════════════

def download_season(
    league_key: str,
    season: str,
    force_refresh: bool = False,
) -> Path | None:
    """
    Download one season's CSV for one league from football-data.co.uk.
    Skips the download if the file already exists locally (caching).

    Args:
        league_key:    Key from config.LEAGUES dict (e.g. 'EPL', 'LA_LIGA')
        season:        Season code (e.g. '2425')
        force_refresh: If True, re-download even if file exists locally

    Returns:
        Path to the saved CSV file, or None if download failed.
    """
    league_info = LEAGUES[league_key]
    fd_code     = league_info["fd_code"]
    url         = build_fd_url(fd_code, season)

    # Where to save: data/raw/EPL_2425.csv
    save_path   = PATHS.RAW / f"{league_key}_{season}.csv"

    # ── Cache check: skip if file already exists ───────────────────────────
    if save_path.exists() and not force_refresh:
        logger.debug(f"Cache hit: {save_path.name} — skipping download")
        return save_path

    # ── Download with retry logic ──────────────────────────────────────────
    headers  = SCRAPING.DEFAULT_HEADERS
    attempts = 0

    while attempts < SCRAPING.MAX_RETRIES:
        try:
            logger.info(f"Downloading {league_info['name']} {SEASONS.to_label(season)} ...")
            time.sleep(SCRAPING.SLEEP_BETWEEN_REQUESTS)   # Respect rate limits

            response = requests.get(
                url,
                headers=headers,
                timeout=SCRAPING.REQUEST_TIMEOUT,
            )

            # football-data.co.uk returns 200 with empty body if season doesn't exist yet
            if response.status_code == 200 and len(response.content) > 100:
                save_path.write_bytes(response.content)
                logger.success(f"Saved: {save_path.name} ({len(response.content):,} bytes)")
                return save_path

            elif response.status_code == 200 and len(response.content) <= 100:
                # Season exists in URL but has no data yet (future season)
                logger.warning(f"Empty response for {league_key} {season} — season may not exist yet")
                return None

            else:
                logger.warning(f"HTTP {response.status_code} for {url}")
                attempts += 1

        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed ({attempts + 1}/{SCRAPING.MAX_RETRIES}): {e}")
            attempts += 1
            time.sleep(SCRAPING.SLEEP_BETWEEN_REQUESTS * 2)  # Back off on error

    logger.error(f"Failed to download after {SCRAPING.MAX_RETRIES} attempts: {url}")
    return None


# ═══════════════════════════════════════════════════════
# SECTION 3 — RAW CSV PARSER
# ═══════════════════════════════════════════════════════

def parse_raw_csv(
    csv_path: Path,
    league_key: str,
    season: str,
) -> pd.DataFrame | None:
    """
    Load and clean one raw CSV file from football-data.co.uk.

    Steps performed:
        1. Load CSV, handle multiple encodings (site uses latin-1 sometimes)
        2. Rename columns using FD_COLUMN_MAP from config
        3. Keep only columns we care about (drop obscure bookmaker columns)
        4. Parse dates correctly
        5. Drop rows missing required fields
        6. Add metadata columns: league_key, season, season_label

    Args:
        csv_path:   Path to the raw CSV file
        league_key: e.g. 'EPL'
        season:     e.g. '2425'

    Returns:
        Cleaned DataFrame, or None if the file is unreadable/empty.
    """
    try:
        # Try UTF-8 first, fall back to latin-1 (football-data.co.uk uses both)
        try:
            df = pd.read_csv(csv_path, encoding="utf-8")
        except UnicodeDecodeError:
            df = pd.read_csv(csv_path, encoding="latin-1")

        # Drop completely empty rows (common at end of football-data.co.uk CSVs)
        df.dropna(how="all", inplace=True)

        if df.empty:
            logger.warning(f"Empty CSV after dropping blank rows: {csv_path.name}")
            return None

        logger.debug(f"Raw shape: {df.shape} — {csv_path.name}")

        # ── Step 1: Rename columns we know about ──────────────────────────
        # Only rename columns that exist in this file (older seasons miss some cols)
        rename_map = {k: v for k, v in FD_COLUMN_MAP.items() if k in df.columns}
        df.rename(columns=rename_map, inplace=True)

        # ── Step 2: Drop rows missing critical fields ──────────────────────
        existing_required = [c for c in REQUIRED_COLUMNS if c in df.columns]
        before = len(df)
        df.dropna(subset=existing_required, inplace=True)
        dropped = before - len(df)
        if dropped > 0:
            logger.debug(f"Dropped {dropped} rows missing required fields in {csv_path.name}")

        if df.empty:
            logger.warning(f"No valid rows remain in {csv_path.name}")
            return None

        # ── Step 3: Parse date column ──────────────────────────────────────
        # football-data.co.uk uses DD/MM/YY or DD/MM/YYYY — handle both
        df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")

        # Drop rows where date parsing failed (malformed rows at bottom of some files)
        date_failures = df["date"].isna().sum()
        if date_failures > 0:
            logger.debug(f"Dropping {date_failures} rows with unparseable dates")
        df.dropna(subset=["date"], inplace=True)

        # ── Step 4: Cast numeric columns safely ───────────────────────────
        numeric_cols = [
            "home_goals", "away_goals",
            "ht_home_goals", "ht_away_goals",
            "home_shots", "away_shots",
            "home_shots_ot", "away_shots_ot",
            "home_corners", "away_corners",
            "home_yellows", "away_yellows",
            "home_reds", "away_reds",
            "odds_home", "odds_draw", "odds_away",
            "odds_over_25", "odds_under_25",
        ]
        for col in numeric_cols:
            if col in df.columns:
                # errors='coerce' turns invalid strings into NaN instead of crashing
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # ── Step 5: Standardise result column ─────────────────────────────
        if "result" in df.columns:
            df["result"] = df["result"].str.strip().str.upper()
            # Keep only valid results — filter out any data corruption
            valid_results = df["result"].isin(["H", "D", "A"])
            invalid_count = (~valid_results).sum()
            if invalid_count > 0:
                logger.debug(f"Dropping {invalid_count} rows with invalid result values")
            df = df[valid_results].copy()

        # Standardise ht_result to string (some seasons use numeric codes)
        if "ht_result" in df.columns:
            df["ht_result"] = df["ht_result"].astype("string").str.strip().str.upper()
            df["ht_result"] = df["ht_result"].replace({"NAN": None, "NONE": None})

        # Standardise referee to string (some seasons have numeric codes)
        if "referee" in df.columns:
            df["referee"] = df["referee"].astype("string").str.strip()
            df["referee"] = df["referee"].replace({"NAN": None, "NONE": None})

        # ── Step 6: Add metadata columns ──────────────────────────────────
        df["league_key"]    = league_key
        df["season"]        = season
        df["season_label"]  = SEASONS.to_label(season)
        df["league_name"]   = LEAGUES[league_key]["name"]
        df["country"]       = LEAGUES[league_key]["country"]

        # Sort chronologically within this file
        df.sort_values("date", inplace=True)
        df.reset_index(drop=True, inplace=True)

        logger.debug(f"Parsed: {len(df)} matches — {csv_path.name}")
        return df

    except Exception as e:
        logger.error(f"Failed to parse {csv_path.name}: {e}")
        return None


# ═══════════════════════════════════════════════════════
# SECTION 3b — ODDS FALLBACK STRATEGY
# ═══════════════════════════════════════════════════════

def fill_missing_odds(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing odds using multi-level fallback strategy.
    
    Priority order:
    1. Bet365 odds (B365H, B365D, B365A) — primary source
    2. Pinnacle odds (PSH, PSD, PSA) — if B365 missing
    3. Betwin odds (BWH, BWD, BWA) — if both missing
    4. Per-league average — if all bookmaker odds missing
    
    This ensures maximum odds coverage while maintaining data quality.
    
    Args:
        df: DataFrame with odds columns (may have NaNs)
        
    Returns:
        DataFrame with filled odds columns
    """
    df = df.copy()
    odds_cols = ["odds_home", "odds_draw", "odds_away"]
    
    # ── Strategy 1: Fill B365 with Pinnacle where missing ──────────────
    if "odds_pinnacle_home" in df.columns:
        for b365_col, pinnacle_col in [
            ("odds_home", "odds_pinnacle_home"),
            ("odds_draw", "odds_pinnacle_draw"),
            ("odds_away", "odds_pinnacle_away"),
        ]:
            if b365_col in df.columns:
                mask = df[b365_col].isna() & df[pinnacle_col].notna() & (df[pinnacle_col] > 1.0)
                filled = mask.sum()
                df.loc[mask, b365_col] = df.loc[mask, pinnacle_col]
                if filled > 0:
                    logger.debug(f"Filled {filled} {b365_col} from Pinnacle")
    
    # ── Strategy 2: Fill with Betwin if still missing ──────────────────
    if "odds_betwin_home" in df.columns:
        for b365_col, betwin_col in [
            ("odds_home", "odds_betwin_home"),
            ("odds_draw", "odds_betwin_draw"),
            ("odds_away", "odds_betwin_away"),
        ]:
            if b365_col in df.columns:
                mask = df[b365_col].isna() & df[betwin_col].notna() & (df[betwin_col] > 1.0)
                filled = mask.sum()
                df.loc[mask, b365_col] = df.loc[mask, betwin_col]
                if filled > 0:
                    logger.debug(f"Filled {filled} {b365_col} from Betwin")
    
    # ── Strategy 3: League average if still missing ────────────────────
    for col in odds_cols:
        if col in df.columns:
            missing_count = df[col].isna().sum()
            if missing_count > 0:
                # Calculate per-league average (only valid odds > 1.0)
                for league in df["league_key"].unique():
                    league_df = df[df["league_key"] == league]
                    league_avg = league_df[league_df[col] > 1.0][col].mean()
                    
                    if not pd.isna(league_avg) and league_avg > 1.0:
                        league_mask = (df["league_key"] == league) & (df[col].isna())
                        filled = league_mask.sum()
                        df.loc[league_mask, col] = league_avg
                        if filled > 0:
                            logger.debug(
                                f"Filled {filled} {col} for {league} "
                                f"with league average ({league_avg:.2f})"
                            )
    
    return df


# ═══════════════════════════════════════════════════════
# SECTION 4 — TARGET VARIABLE DERIVATION
# ═══════════════════════════════════════════════════════

def add_target_variables(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive all prediction target variables from raw match data.
    These are the columns our models will learn to predict.

    Targets added:
        - btts        : 1 if both teams scored, else 0
        - over_25     : 1 if total goals > 2.5, else 0
        - total_goals : integer total goals in match
        - total_corners: integer total corners (if data available)
        - over_corners: 1 if total corners > 9.5, else 0
        - result_numeric: H=1, D=0, A=2 (for stratification checks)

    Args:
        df: DataFrame with at least home_goals and away_goals columns

    Returns:
        DataFrame with new target columns added in-place.
    """
    df = df.copy()

    # ── Both Teams To Score (BTTS) ─────────────────────────────────────────
    # Yes (1) if both home_goals >= 1 AND away_goals >= 1
    if "home_goals" in df.columns and "away_goals" in df.columns:
        df["btts"] = (
            (df["home_goals"] >= 1) & (df["away_goals"] >= 1)
        ).astype(int)

    # ── Total goals and Over 2.5 ───────────────────────────────────────────
        df["total_goals"] = df["home_goals"] + df["away_goals"]
        df["over_25"]     = (df["total_goals"] > 2.5).astype(int)

    # ── Total corners and Over 9.5 ────────────────────────────────────────
    if "home_corners" in df.columns and "away_corners" in df.columns:
        df["total_corners"] = df["home_corners"] + df["away_corners"]
        df["over_corners"]  = (df["total_corners"] > 9.5).astype(int)

    # ── Numeric result encoding ────────────────────────────────────────────
    # Useful for stratified splits and some sklearn utilities
    if "result" in df.columns:
        result_map = {"H": 1, "D": 0, "A": 2}
        df["result_numeric"] = df["result"].map(result_map)

    # ── Goal difference ────────────────────────────────────────────────────
    if "home_goals" in df.columns and "away_goals" in df.columns:
        df["goal_diff"] = df["home_goals"] - df["away_goals"]

    return df


# ═══════════════════════════════════════════════════════
# SECTION 5 — BOOKMAKER MARGIN REMOVAL
# ═══════════════════════════════════════════════════════

def add_fair_odds(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove the bookmaker's overround to get the implied 'fair' probability
    for each outcome. This is used by the betting engine to identify value bets.

    The overround is the sum of implied probabilities (1/odds) across all outcomes.
    A fair market would sum to 1.0; bookmakers inflate it to ~1.05-1.10 for margin.

    We remove the margin by dividing each implied probability by the total overround,
    normalising them back to sum to 1.0.

    Args:
        df: DataFrame with odds_home, odds_draw, odds_away columns

    Returns:
        DataFrame with added columns:
            - implied_prob_home / draw / away  (raw implied probabilities)
            - overround                         (total book margin)
            - fair_prob_home / draw / away      (margin-removed fair probabilities)
    """
    df = df.copy()

    odds_cols = ["odds_home", "odds_draw", "odds_away"]
    # Only process rows where all three odds are present and valid (> 1.0)
    mask = (
        df[odds_cols].notna().all(axis=1) &
        (df[odds_cols] > 1.0).all(axis=1)
    )

    if mask.sum() == 0:
        logger.debug("No valid odds found for fair odds calculation")
        return df

    # Raw implied probabilities (1 / decimal odds)
    df.loc[mask, "implied_prob_home"] = 1 / df.loc[mask, "odds_home"]
    df.loc[mask, "implied_prob_draw"] = 1 / df.loc[mask, "odds_draw"]
    df.loc[mask, "implied_prob_away"] = 1 / df.loc[mask, "odds_away"]

    # Overround = sum of implied probs (> 1.0 means bookmaker margin)
    df.loc[mask, "overround"] = (
        df.loc[mask, "implied_prob_home"] +
        df.loc[mask, "implied_prob_draw"] +
        df.loc[mask, "implied_prob_away"]
    )

    # Fair probabilities = implied / overround (removes the margin)
    df.loc[mask, "fair_prob_home"] = df.loc[mask, "implied_prob_home"] / df.loc[mask, "overround"]
    df.loc[mask, "fair_prob_draw"] = df.loc[mask, "implied_prob_draw"] / df.loc[mask, "overround"]
    df.loc[mask, "fair_prob_away"] = df.loc[mask, "implied_prob_away"] / df.loc[mask, "overround"]

    logger.debug(f"Fair odds calculated for {mask.sum():,} matches. "
                 f"Avg overround: {df.loc[mask, 'overround'].mean():.4f}")
    return df


# ═══════════════════════════════════════════════════════
# SECTION 6 — MASTER DOWNLOAD ORCHESTRATOR
# ═══════════════════════════════════════════════════════

def download_all(
    leagues: list[str] | None = None,
    seasons: list[str] | None = None,
    force_refresh: bool = False,
) -> dict[str, Path]:
    """
    Download CSVs for all specified leagues and seasons.

    Args:
        leagues:       List of league keys to download (default: all 5)
        seasons:       List of season codes to download (default: all configured)
        force_refresh: Re-download even if cached

    Returns:
        Dict mapping '{league_key}_{season}' → Path to downloaded CSV
        (None value means download failed for that combo)
    """
    leagues = leagues or list(LEAGUES.keys())
    seasons = seasons or SEASONS.ALL_SEASONS

    total   = len(leagues) * len(seasons)
    results = {}

    logger.info(f"Starting download: {len(leagues)} leagues × {len(seasons)} seasons = {total} files")

    with tqdm(total=total, desc="Downloading", unit="file") as pbar:
        for league_key in leagues:
            for season in seasons:
                key  = f"{league_key}_{season}"
                path = download_season(league_key, season, force_refresh=force_refresh)
                results[key] = path
                pbar.update(1)
                pbar.set_postfix({"last": key, "ok": sum(v is not None for v in results.values())})

    successful = sum(v is not None for v in results.values())
    logger.info(f"Download complete: {successful}/{total} files successfully obtained")
    return results


# ═══════════════════════════════════════════════════════
# SECTION 7 — BUILD MASTER DATAFRAME
# ═══════════════════════════════════════════════════════

def build_master_df(
    leagues: list[str] | None = None,
    seasons: list[str] | None = None,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Full pipeline: download → parse → derive targets → merge → return master DataFrame.

    This is the main entry point for building the dataset from scratch.

    Args:
        leagues:       Leagues to include (default: all 5)
        seasons:       Seasons to include (default: all configured)
        force_refresh: Re-download all files even if cached

    Returns:
        Master DataFrame with all matches, standardised columns, and target variables.
        Sorted chronologically across all leagues.
    """
    leagues = leagues or list(LEAGUES.keys())
    seasons = seasons or SEASONS.ALL_SEASONS

    # ── Step 1: Download all raw CSVs ─────────────────────────────────────
    download_all(leagues=leagues, seasons=seasons, force_refresh=force_refresh)

    # ── Step 2: Parse each downloaded CSV ─────────────────────────────────
    all_frames = []
    total      = len(leagues) * len(seasons)

    logger.info("Parsing downloaded CSV files ...")

    with tqdm(total=total, desc="Parsing CSVs", unit="file") as pbar:
        for league_key in leagues:
            for season in seasons:
                csv_path = PATHS.RAW / f"{league_key}_{season}.csv"

                if not csv_path.exists():
                    logger.debug(f"Skipping missing file: {csv_path.name}")
                    pbar.update(1)
                    continue

                df = parse_raw_csv(csv_path, league_key, season)

                if df is not None and not df.empty:
                    all_frames.append(df)

                pbar.update(1)

    if not all_frames:
        logger.error("No data was successfully parsed. Check internet connection and rerun.")
        raise RuntimeError("No valid data frames were built.")

    # ── Step 3: Concatenate all leagues and seasons ────────────────────────
    logger.info(f"Merging {len(all_frames)} DataFrames ...")
    master = pd.concat(all_frames, ignore_index=True)

    # ── Step 4: Sort chronologically (critical — never shuffle football data) ──
    master.sort_values(["date", "league_key"], inplace=True)
    master.reset_index(drop=True, inplace=True)

    # Drop ht_result if present (mixed types across seasons, not used as a feature)
    if "ht_result" in master.columns:
        master.drop(columns=["ht_result"], inplace=True)

    # ── Step 5: Derive target variables ───────────────────────────────────
    logger.info("Deriving target variables (BTTS, Over 2.5, Corners) ...")
    master = add_target_variables(master)

    # ── Step 5b: Fill missing odds with fallback strategy ────────────────
    logger.info("Filling missing odds with fallback strategy ...")
    master = fill_missing_odds(master)

    # ── Step 6: Add fair odds (remove bookmaker margin) ───────────────────
    logger.info("Calculating fair odds (removing bookmaker overround) ...")
    master = add_fair_odds(master)

    # ── Step 7: Final data quality report ─────────────────────────────────
    _log_data_quality(master)

    logger.success(f"Master DataFrame built: {len(master):,} matches, {master.shape[1]} columns")
    return master


# ═══════════════════════════════════════════════════════
# SECTION 8 — SAVE & LOAD HELPERS
# ═══════════════════════════════════════════════════════

def save_master_df(df: pd.DataFrame) -> None:
    """
    Save master DataFrame to parquet (fast) and per-league splits.

    Args:
        df: The master DataFrame returned by build_master_df()
    """
    # ── Save full master ───────────────────────────────────────────────────
    master_path = PATHS.PROCESSED / "master.parquet"
    # FIX: Pylance stubs missing 'pyarrow' — suppress false positive
    df.to_parquet(master_path, index=False, engine="pyarrow")  # type: ignore[call-overload]
    logger.success(f"Saved master: {master_path} ({master_path.stat().st_size / 1024:.1f} KB)")

    # ── Save per-league files (convenience for league-specific analysis) ───
    for league_key in df["league_key"].unique():
        league_df   = df[df["league_key"] == league_key].copy()
        league_path = PATHS.PROCESSED / f"{league_key}_master.parquet"
        league_df.to_parquet(league_path, index=False, engine="pyarrow")  # type: ignore[call-overload]
        logger.debug(f"Saved {league_key}: {len(league_df):,} matches → {league_path.name}")


def load_master_df(league_key: str | None = None) -> pd.DataFrame:
    """
    Load the pre-processed master DataFrame from parquet.
    Call this in notebooks and downstream modules instead of re-running build_master_df().

    Args:
        league_key: If provided, load only that league (e.g. 'EPL').
                    If None, load all leagues.

    Returns:
        DataFrame ready for feature engineering.

    Raises:
        FileNotFoundError: If parquet file doesn't exist yet — run build_master_df() first.
    """
    if league_key:
        path = PATHS.PROCESSED / f"{league_key}_master.parquet"
    else:
        path = PATHS.PROCESSED / "master.parquet"

    if not path.exists():
        raise FileNotFoundError(
            f"Parquet file not found: {path}\n"
            "Run: python src/data_loader.py    to build the dataset first."
        )

    # FIX: Pylance stubs missing 'pyarrow' — suppress false positive
    df = pd.read_parquet(path, engine="pyarrow")  # type: ignore[call-overload]
    logger.info(f"Loaded: {len(df):,} matches from {path.name}")
    return df


# ═══════════════════════════════════════════════════════
# SECTION 9 — DATA QUALITY REPORT
# ═══════════════════════════════════════════════════════

def _log_data_quality(df: pd.DataFrame) -> None:
    """
    Print a human-readable summary of the master DataFrame.
    Called automatically by build_master_df() after merging.
    """
    logger.info("=" * 60)
    logger.info("DATA QUALITY REPORT")
    logger.info("=" * 60)
    logger.info(f"Total matches      : {len(df):,}")
    logger.info(f"Date range         : {df['date'].min().date()} → {df['date'].max().date()}")
    logger.info(f"Columns            : {df.shape[1]}")
    logger.info(f"Leagues            : {df['league_key'].nunique()}")
    logger.info(f"Seasons            : {sorted(df['season'].unique())}")
    logger.info("")

    # Per-league breakdown
    for league_key in sorted(df["league_key"].unique()):
        sub   = df[df["league_key"] == league_key]
        teams = sub["home_team"].nunique()
        logger.info(
            f"  {league_key:<12} {len(sub):>5,} matches | "
            f"{teams:>2} teams | "
            f"{sub['season'].nunique()} seasons"
        )

    logger.info("")

    # Missing data summary — only show columns with missing values
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    if not missing.empty:
        logger.info("Columns with missing values:")
        for col, count in missing.items():
            pct = count / len(df) * 100
            logger.info(f"  {col:<25} {count:>6,} ({pct:.1f}%)")
    else:
        logger.info("No missing values in required columns.")

    logger.info("")

    # Target variable distributions
    if "result" in df.columns:
        counts = df["result"].value_counts()
        total  = len(df)
        logger.info("Result distribution:")
        for label, val in [("Home Win (H)", "H"), ("Draw (D)", "D"), ("Away Win (A)", "A")]:
            n   = counts.get(val, 0)
            pct = n / total * 100
            logger.info(f"  {label:<18} {n:>6,} ({pct:.1f}%)")

    if "btts" in df.columns:
        btts_rate = df["btts"].mean() * 100
        logger.info(f"BTTS Yes rate      : {btts_rate:.1f}%")

    if "over_25" in df.columns:
        over_rate = df["over_25"].mean() * 100
        logger.info(f"Over 2.5 rate      : {over_rate:.1f}%")

    if "odds_home" in df.columns:
        odds_coverage = df["odds_home"].notna().mean() * 100
        logger.info(f"Bet365 odds coverage: {odds_coverage:.1f}%")

    # Report on fallback odds sources
    if "odds_pinnacle_home" in df.columns:
        pinnacle_coverage = df["odds_pinnacle_home"].notna().mean() * 100
        logger.info(f"Pinnacle odds coverage (backup): {pinnacle_coverage:.1f}%")
    
    if "odds_betwin_home" in df.columns:
        betwin_coverage = df["odds_betwin_home"].notna().mean() * 100
        logger.info(f"Betwin odds coverage (backup): {betwin_coverage:.1f}%")

    logger.info("=" * 60)


def print_data_quality_report(df: pd.DataFrame | None = None) -> None:
    """
    Public wrapper to print the data quality report.
    If df is None, loads master from parquet first.
    """
    if df is None:
        df = load_master_df()
    _log_data_quality(df)


# ═══════════════════════════════════════════════════════
# MAIN — run as script to download and process everything
# ═══════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Download and process football data from football-data.co.uk"
    )
    parser.add_argument(
        "--leagues",
        nargs="+",
        choices=list(LEAGUES.keys()),
        default=None,
        help="Leagues to download (default: all 5). E.g. --leagues EPL LA_LIGA",
    )
    parser.add_argument(
        "--seasons",
        nargs="+",
        default=None,
        help="Seasons to download (default: all). E.g. --seasons 2324 2425",
    )
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Re-download all files even if cached locally",
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Skip downloading — just print data quality report on existing parquet",
    )
    args = parser.parse_args()

    # Ensure all directories exist before doing anything
    PATHS.create_all()

    if args.report_only:
        print_data_quality_report()
    else:
        # Full pipeline: download → parse → merge → save
        master_df = build_master_df(
            leagues=args.leagues,
            seasons=args.seasons,
            force_refresh=args.force_refresh,
        )
        save_master_df(master_df)
        logger.success("All done. Run notebook 01_data_exploration.ipynb to explore the data.")