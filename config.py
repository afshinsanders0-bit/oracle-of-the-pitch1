"""
config.py — Central configuration for the Football Prediction Platform
=======================================================================
PURPOSE:
    Single source of truth for every constant in the project.
    All other modules import from here — NOTHING is hardcoded elsewhere.

USAGE:
    from config import LEAGUES, PATHS, MODEL_PARAMS, KELLY

AUTHOR:  Football Predictor Platform
VERSION: 1.1.0 (Added Champions League support)
"""

from pathlib import Path

# ═══════════════════════════════════════════════════════
# ROOT PATHS  — all paths derived from project root
# ═══════════════════════════════════════════════════════

# Absolute path to this config file's directory (project root)
ROOT_DIR = Path(__file__).parent.resolve()

class PATHS:
    """
    All file system paths used across the project.
    Uses pathlib.Path throughout — never use hardcoded strings.
    """
    # Top-level directories
    DATA        = ROOT_DIR / "data"
    NOTEBOOKS   = ROOT_DIR / "notebooks"
    SRC         = ROOT_DIR / "src"
    MODELS      = ROOT_DIR / "models"

    # Data subdirectories
    RAW         = DATA / "raw"          # Downloaded CSVs per league per season
    PROCESSED   = DATA / "processed"    # Feature-engineered parquet files
    LIVE        = DATA / "live"         # Live match JSON cache
    CACHE       = DATA / "cache"        # Scraped HTML cache (avoids re-fetching)

    # Model save paths (one .pkl per market)
    MODEL_MATCH_RESULT  = MODELS / "match_result.pkl"
    MODEL_BTTS          = MODELS / "btts.pkl"
    MODEL_OVER_UNDER    = MODELS / "over_under.pkl"
    MODEL_CORNERS       = MODELS / "corners.pkl"
    MODEL_DIXON_COLES   = MODELS / "dixon_coles.pkl"

    # Log file
    LOG_FILE    = ROOT_DIR / "predictor.log"

    @classmethod
    def create_all(cls) -> None:
        """Create all directories if they don't exist. Call once on startup."""
        for attr_name in dir(cls):
            attr = getattr(cls, attr_name)
            # Only create directories (Path objects that don't have a suffix = not files)
            if isinstance(attr, Path) and not attr.suffix:
                attr.mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════════════
# LEAGUE CONFIGURATION
# ═══════════════════════════════════════════════════════

LEAGUES = {
    # ── DOMESTIC LEAGUES ──────────────────────────────────────────────────
    "EPL": {
        "name":         "English Premier League",
        "fd_code":      "E0",           # football-data.co.uk league code
        "country":      "England",
        "elo_country":  "England",      # ClubElo.com country label
        "fd_tier":      1,              # Tier 1 = top flight
        "is_european":  False,
    },
    "ENG_CHAMP": {
        "name":         "English Championship",
        "fd_code":      "E1",
        "country":      "England",
        "elo_country":  "England",
        "fd_tier":      2,
        "is_european":  False,
    },
    "LA_LIGA": {
        "name":         "La Liga",
        "fd_code":      "SP1",
        "country":      "Spain",
        "elo_country":  "Spain",
        "fd_tier":      1,
        "is_european":  False,
    },
    "SERIE_A": {
        "name":         "Serie A",
        "fd_code":      "I1",
        "country":      "Italy",
        "elo_country":  "Italy",
        "fd_tier":      1,
        "is_european":  False,
    },
    "LIGUE_1": {
        "name":         "Ligue 1",
        "fd_code":      "F1",
        "country":      "France",
        "elo_country":  "France",
        "fd_tier":      1,
        "is_european":  False,
    },

    # ── EUROPEAN COMPETITIONS ─────────────────────────────────────────────
    "CHAMPIONS_LEAGUE": {
        "name":         "UEFA Champions League",
        "fd_code":      "CL",           # football-data.co.uk code for CL
        "country":      "European",     # Multi-country competition
        "elo_country":  "International", # ClubElo treats as international pool
        "fd_tier":      0,              # Tier 0 = continental (not domestic league)
        "is_european":  True,           # Flag for CL-specific feature engineering
        "stage":        "group",        # group / round_of_16 / quarter / semi / final
        "home_advantage_reduced": True, # CL has less home advantage than domestic
    },
}

# All league codes as a flat list (used for iteration)
ALL_LEAGUE_CODES = list(LEAGUES.keys())

# Domestic leagues only (for some analyses)
DOMESTIC_LEAGUE_CODES = [k for k, v in LEAGUES.items() if not v.get("is_european", False)]

# European competitions only
EUROPEAN_LEAGUE_CODES = [k for k, v in LEAGUES.items() if v.get("is_european", False)]


# ═══════════════════════════════════════════════════════
# DATA SOURCE URLs
# ═══════════════════════════════════════════════════════

class URLS:
    """Base URLs for every data source. Never hardcode these in scrapers."""

    # football-data.co.uk — primary source for results, odds, corners, cards, shots
    # Final URL pattern: {BASE_FD}/{season}/{fd_code}.csv
    # e.g. https://www.football-data.co.uk/mmz4281/2425/E0.csv (domestic)
    #      https://www.football-data.co.uk/mmz4281/2425/CL.csv (Champions League)
    BASE_FD         = "https://www.football-data.co.uk/mmz4281"

    # ClubElo.com — ELO ratings REST API (returns CSV)
    # Final URL: {CLUBELO_TEAM}/{team_name}  →  returns all historical ELO for that team
    CLUBELO_TEAM    = "http://api.clubelo.com"
    CLUBELO_DATE    = "http://api.clubelo.com/{date}"  # All team ELOs on a given date

    # Understat.com — xG data per league per season
    # Includes: EPL, La Liga, Serie A, Ligue 1, Bundesliga (NOT CL)
    UNDERSTAT_BASE  = "https://understat.com/league"
    UNDERSTAT_CL    = "https://understat.com/team"  # CL uses team-specific pages

    # FBref.com — deep stats, PPDA pressing data
    # Has CL stats in their European Competitions section
    FBREF_BASE      = "https://fbref.com/en/comps"
    FBREF_CL        = "https://fbref.com/en/comps/8"  # Competition ID 8 = Champions League

    # Sofascore — live in-play stats (requires Selenium — JS rendered)
    SOFASCORE_BASE  = "https://www.sofascore.com"

    # Transfermarkt — squad values and injury news (covers all leagues + CL)
    TRANSFERMARKT_BASE = "https://www.transfermarkt.com"

    # Worldfootball.net — referee statistics
    WORLDFOOTBALL_BASE = "https://www.worldfootball.net"


# ═══════════════════════════════════════════════════════
# SEASON CONFIGURATION
# ═══════════════════════════════════════════════════════

class SEASONS:
    """
    Seasons used for training, validation, and holdout testing.
    football-data.co.uk uses 4-digit codes like '1819' for 2018-19.
    
    NOTE: Champions League seasons align with domestic leagues (2024-25 CL runs 24/25 season).
    """
    # Full training history available (domestic leagues)
    ALL_SEASONS = [
        "1819", "1920", "2021", "2122", "2223", "2324", "2425"
    ]

    # CL-specific season availability (may be shorter history on football-data.co.uk)
    CL_SEASONS = ["2223", "2324", "2425"]  # Adjust if earlier CL data becomes available

    # Train / validate / test split — NEVER shuffle, always chronological
    TRAIN       = ["1819", "1920", "2021", "2122", "2223"]  # 2018-19 to 2022-23
    VALIDATION  = ["2324"]                                   # 2023-24 season
    TEST        = ["2425"]                                   # 2024-25 (never touched during training)

    # Current live season (update each August when new season starts)
    CURRENT     = "2425"

    @staticmethod
    def to_label(code: str) -> str:
        """
        Convert season code to human-readable label.
        e.g. '2425' → '2024-25'
        """
        return f"20{code[:2]}-{code[2:]}"


# ═══════════════════════════════════════════════════════
# football-data.co.uk COLUMN MAPPING
# ═══════════════════════════════════════════════════════

# Standardised column names we rename raw fd.co.uk columns to.
# This isolates us from any upstream renaming on their end.
# Works for both domestic leagues and Champions League matches.
FD_COLUMN_MAP = {
    "Div":    "league",
    "Date":   "date",
    "HomeTeam": "home_team",
    "AwayTeam": "away_team",
    "FTHG":   "home_goals",      # Full-time home goals
    "FTAG":   "away_goals",      # Full-time away goals
    "FTR":    "result",          # H / D / A
    "HTHG":   "ht_home_goals",   # Half-time home goals
    "HTAG":   "ht_away_goals",   # Half-time away goals
    "HTR":    "ht_result",       # Half-time result
    "HS":     "home_shots",
    "AS":     "away_shots",
    "HST":    "home_shots_ot",   # Shots on target
    "AST":    "away_shots_ot",
    "HC":     "home_corners",
    "AC":     "away_corners",
    "HY":     "home_yellows",
    "AY":     "away_yellows",
    "HR":     "home_reds",
    "AR":     "away_reds",
    "Referee": "referee",
    # Bet365 closing odds (most liquid market)
    "B365H":  "odds_home",
    "B365D":  "odds_draw",
    "B365A":  "odds_away",
    # Over/Under 2.5
    "B365>2.5": "odds_over_25",
    "B365<2.5": "odds_under_25",
}

# Minimum columns required after renaming — rows missing any of these are dropped
REQUIRED_COLUMNS = [
    "date", "home_team", "away_team",
    "home_goals", "away_goals", "result",
]


# ═══════════════════════════════════════════════════════
# FEATURE ENGINEERING PARAMETERS
# ═══════════════════════════════════════════════════════

class FEATURES:
    """Parameters controlling feature creation in feature_engineering.py."""

    # Rolling window sizes for form averages
    ROLLING_WINDOWS = [3, 5, 10]

    # Primary window used when a single window is needed (e.g. xG rolling avg)
    PRIMARY_WINDOW = 5

    # Minimum matches a team must have played before we calculate rolling features.
    # Rows below this threshold will have NaN features (handled in pipeline).
    MIN_MATCHES_FOR_ROLLING = 3

    # ELO starting rating for any team with no prior history
    ELO_DEFAULT_RATING = 1500

    # Target variable names — must match model training expectations
    TARGET_MATCH_RESULT = "result"          # Values: H / D / A
    TARGET_BTTS         = "btts"            # Values: 1 (yes) / 0 (no)
    TARGET_OVER_UNDER   = "over_25"         # Values: 1 (over) / 0 (under)
    TARGET_CORNERS      = "over_corners"    # Values: 1 (over 9.5) / 0 (under)

    # Corner market line
    CORNER_LINE = 9.5

    # Goal market line
    GOAL_LINE = 2.5

    # Match importance score thresholds (domestic league position)
    IMPORTANCE_TOP4_CUTOFF       = 4   # Within top 4 positions
    IMPORTANCE_RELEGATION_CUTOFF = 3   # Bottom 3 positions (standard 20-team league)

    # Fatigue index — days rest thresholds
    FATIGUE_RESTED_DAYS     = 6    # 6+ days rest = fully rested
    FATIGUE_TIRED_DAYS      = 3    # 3 or fewer days = fatigue flag raised
    FATIGUE_WINDOW_DAYS     = 30   # Count matches in last 30 days for congestion

    # ── CL-SPECIFIC FEATURES ──────────────────────────────────────────────
    # Champions League-specific feature engineering flags
    USE_CL_STAGE_ENCODING   = True      # Encode CL stage (group / knockout)
    USE_CL_DRAW_ENCODING    = True      # Encode CL draw importance (seeding)
    USE_CL_EXPERIENCE       = True      # Team's historical CL appearance rate
    REDUCE_CL_HOME_ADVANTAGE = True     # CL has ~70 ELO home advantage vs 100 domestic


# ═══════════════════════════════════════════════════════
# ELO SYSTEM PARAMETERS
# ═══════════════════════════════════════════════════════

class ELO:
    """
    Parameters for the dynamic ELO rating system in elo.py.
    ELO updates after every match result.
    
    NOTE: Champions League teams share a global ELO pool (not separated by country).
    Domestic league ELO is tracked separately per league.
    """
    # Starting ELO for all teams before any data
    DEFAULT_RATING      = 1500

    # K-factor controls how fast ratings change.
    # Higher K = faster response to results, but noisier.
    K_FACTOR            = 32

    # K-factor multiplier for CL (CL matches are more informative than domestic)
    CL_K_FACTOR_MULTIPLIER = 1.2  # 1.2x normal K-factor for CL

    # Home advantage built into the ELO expectation calculation
    # Typical football home advantage is ~100 ELO points (domestic)
    HOME_ADVANTAGE      = 100

    # CL home advantage is slightly reduced (~70 ELO) due to:
    # - Neutral venues in finals
    # - Less familiar stadiums (not home ground)
    # - Higher quality opposition reduces home effect
    CL_HOME_ADVANTAGE   = 70

    # When a promoted team enters a higher division, regress their ELO
    # toward the league mean by this factor (e.g. 0.5 = halfway to mean)
    PROMOTION_REGRESSION = 0.5


# ═══════════════════════════════════════════════════════
# MODEL PARAMETERS (defaults before Optuna tuning)
# ═══════════════════════════════════════════════════════

class MODEL_PARAMS:
    """
    Default hyperparameters for each model.
    Optuna will search around these — treat as starting point, not final values.
    """

    # TimeSeriesSplit — always used, never shuffle
    TS_SPLIT_N_FOLDS = 5

    XGBOOST_DEFAULT = {
        "n_estimators":     500,
        "max_depth":        5,
        "learning_rate":    0.05,
        "subsample":        0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 3,
        "use_label_encoder": False,
        "eval_metric":      "mlogloss",
        "random_state":     42,
        "n_jobs":           -1,
    }

    LIGHTGBM_DEFAULT = {
        "n_estimators":     500,
        "max_depth":        5,
        "learning_rate":    0.05,
        "subsample":        0.8,
        "colsample_bytree": 0.8,
        "num_leaves":       31,
        "random_state":     42,
        "n_jobs":           -1,
        "verbose":          -1,         # Suppress LightGBM output spam
    }

    RANDOM_FOREST_DEFAULT = {
        "n_estimators":     300,
        "max_depth":        8,
        "min_samples_leaf": 5,
        "random_state":     42,
        "n_jobs":           -1,
    }

    LOGISTIC_DEFAULT = {
        "C":                1.0,
        "max_iter":         1000,
        "random_state":     42,
        "solver":           "lbfgs",
        "multi_class":      "auto",
    }

    # Dixon-Coles Poisson correction parameter (rho)
    # Typical literature value: around -0.13
    DIXON_COLES_RHO_INIT = -0.13

    # Number of top scorelines Dixon-Coles returns
    DIXON_COLES_TOP_N = 10


# ═══════════════════════════════════════════════════════
# SPORTS INVESTING ENGINE PARAMETERS
# ═══════════════════════════════════════════════════════

class KELLY:
    """
    Kelly Criterion staking parameters.
    These are the guardrails that make this investing, not gambling.
    """
    # Fractional Kelly multiplier — use 25% of full Kelly to reduce variance
    FRACTION = 0.25

    # Hard cap: never stake more than 5% of bankroll on one bet, regardless of Kelly output
    MAX_STAKE_PCT = 0.05

    # Minimum Kelly fraction before we skip the bet (< 0.5% edge = not worth it)
    MIN_STAKE_PCT = 0.005

    # Default starting bankroll in units
    DEFAULT_BANKROLL = 1000.0


class VALUE_BET:
    """
    Thresholds for flagging a bet as a value bet.
    A bet is only recommended when our edge exceeds these thresholds.
    """
    # Minimum edge over bookmaker's implied probability to flag as value
    # 0.05 = our model probability must be at least 5% higher than fair odds imply
    MIN_EDGE = 0.05

    # Minimum model probability — never bet on outcomes we rate < 30%
    MIN_MODEL_PROB = 0.30

    # Maximum bookmaker odds — avoid longshots (often noise, not signal)
    MAX_ODDS = 8.00

    # Minimum bookmaker odds — skip heavy favourites (low EV ceiling)
    MIN_ODDS = 1.20

    # Confidence tier thresholds
    HIGH_CONFIDENCE   = 0.65    # Model top probability > 65%
    MEDIUM_CONFIDENCE = 0.50    # Model top probability 50-65%
    # Below 0.50 = LOW confidence → do not bet


# ═══════════════════════════════════════════════════════
# MARKET LABELS
# ═══════════════════════════════════════════════════════

MARKETS = {
    "match_result":     "Match Result (1X2)",
    "btts":             "Both Teams to Score",
    "over_under":       "Over/Under 2.5 Goals",
    "corners":          "Corners Over/Under 9.5",
    "dixon_coles":      "Exact Score (Dixon-Coles)",
}

# Human-readable result labels
RESULT_LABELS = {
    "H": "Home Win",
    "D": "Draw",
    "A": "Away Win",
}


# ═══════════════════════════════════════════════════════
# SCRAPING SETTINGS
# ═══════════════════════════════════════════════════════

class SCRAPING:
    """Rate limiting and caching settings for all scrapers."""

    # Seconds to sleep between HTTP requests — respect robots.txt
    SLEEP_BETWEEN_REQUESTS = 2.0

    # Extra sleep when hitting a new domain (first request per session)
    SLEEP_FIRST_REQUEST    = 3.0

    # Maximum retries before giving up on a URL
    MAX_RETRIES            = 3

    # Request timeout in seconds
    REQUEST_TIMEOUT        = 15

    # User-Agent header — identify ourselves honestly
    USER_AGENT = (
        "Mozilla/5.0 (compatible; FootballPredictor/1.0; "
        "research-only; contact: your@email.com)"
    )

    # Headers sent with every request
    DEFAULT_HEADERS = {
        "User-Agent":       USER_AGENT,
        "Accept-Language":  "en-GB,en;q=0.9",
        "Accept-Encoding":  "gzip, deflate, br",
    }

    # Cache expiry — don't re-scrape data younger than this (hours)
    CACHE_EXPIRY_HOURS     = 24

    # Live match scrape interval (seconds) — every 5 minutes during matches
    LIVE_SCRAPE_INTERVAL   = 300


# ═══════════════════════════════════════════════════════
# LOGGING CONFIGURATION
# ═══════════════════════════════════════════════════════

class LOGGING:
    """Loguru logging settings. Applied in each module's setup block."""
    LEVEL   = "INFO"        # Change to "DEBUG" for verbose output during development
    FORMAT  = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{line}</cyan> — "
        "<level>{message}</level>"
    )
    ROTATION   = "10 MB"    # Rotate log file when it hits 10MB
    RETENTION  = "30 days"  # Keep logs for 30 days


# ═══════════════════════════════════════════════════════
# RANDOM SEED (reproducibility)
# ═══════════════════════════════════════════════════════

RANDOM_SEED = 42    # Set everywhere: numpy, sklearn, xgboost, lightgbm, optuna


# ═══════════════════════════════════════════════════════
# STARTUP CHECK — run to verify config is importable
# ═══════════════════════════════════════════════════════

if __name__ == "__main__":
    print("✓ config.py loaded successfully")
    print(f"  Project root : {ROOT_DIR}")
    print(f"  Domestic leagues : {', '.join(DOMESTIC_LEAGUE_CODES)}")
    print(f"  European comps   : {', '.join(EUROPEAN_LEAGUE_CODES)}")
    print(f"  Train seasons: {', '.join(SEASONS.TRAIN)}")
    print(f"  Test season  : {SEASONS.TEST[0]}")
    print(f"  CL seasons   : {', '.join(SEASONS.CL_SEASONS)}")
    print(f"  Kelly fraction: {KELLY.FRACTION * 100:.0f}% of full Kelly")
    print(f"  Value edge min: {VALUE_BET.MIN_EDGE * 100:.0f}%")
    print(f"  CL home advantage: {ELO.CL_HOME_ADVANTAGE} ELO (vs {ELO.HOME_ADVANTAGE} domestic)")
    print()
    print("Creating directories...")
    PATHS.create_all()
    print("✓ All directories created")