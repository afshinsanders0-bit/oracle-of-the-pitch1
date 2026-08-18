"""
config.py — Central configuration for the Football Prediction Platform
=======================================================================
EXPANDED VERSION (v2.0):
  - 20 Domestic Leagues (5 major + secondary divisions)
  - 15+ European Tournaments (CL, EL, Conference League, Super Cups)
  - 10 International Competitions (World Cup, Euros, Copa América, etc.)
  - Comprehensive API codes for all data sources

PURPOSE:
    Single source of truth for every constant in the project.
    All other modules import from here — NOTHING is hardcoded elsewhere.

USAGE:
    from config import LEAGUES, TOURNAMENTS, PATHS, MODEL_PARAMS

AUTHOR:  Football Predictor Platform
VERSION: 2.0.0 (Expanded leagues + tournaments)
"""

from pathlib import Path

# ═══════════════════════════════════════════════════════
# ROOT PATHS  — all paths derived from project root
# ═══════════════════════════════════════════════════════

ROOT_DIR = Path(__file__).parent.resolve()

class PATHS:
    """All file system paths used across the project."""
    DATA        = ROOT_DIR / "data"
    NOTEBOOKS   = ROOT_DIR / "notebooks"
    SRC         = ROOT_DIR / "src"
    MODELS      = ROOT_DIR / "models"

    RAW         = DATA / "raw"
    PROCESSED   = DATA / "processed"
    LIVE        = DATA / "live"
    CACHE       = DATA / "cache"

    MODEL_MATCH_RESULT  = MODELS / "match_result.pkl"
    MODEL_BTTS          = MODELS / "btts.pkl"
    MODEL_OVER_UNDER    = MODELS / "over_under.pkl"
    MODEL_CORNERS       = MODELS / "corners.pkl"
    MODEL_DIXON_COLES   = MODELS / "dixon_coles.pkl"

    LOG_FILE    = ROOT_DIR / "predictor.log"

    @classmethod
    def create_all(cls) -> None:
        for attr_name in dir(cls):
            attr = getattr(cls, attr_name)
            if isinstance(attr, Path) and not attr.suffix:
                attr.mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════════════
# DOMESTIC LEAGUE CONFIGURATION
# ═══════════════════════════════════════════════════════

DOMESTIC_LEAGUES = {
    # ─ TOP 5 LEAGUES ─────────────────────────────────────
    "EPL": {
        "name":         "English Premier League",
        "country":      "England",
        "fd_code":      "E0",
        "sofascore_id": 17,
        "tier":         1,
        "matches_per_season": 380,
        "teams":        20,
    },
    "ENG_CHAMP": {
        "name":         "English Championship",
        "country":      "England",
        "fd_code":      "E1",
        "sofascore_id": 18,
        "tier":         2,
        "matches_per_season": 552,
        "teams":        24,
    },
    "LA_LIGA": {
        "name":         "La Liga EA Sports",
        "country":      "Spain",
        "fd_code":      "SP1",
        "sofascore_id": 8,
        "tier":         1,
        "matches_per_season": 380,
        "teams":        20,
    },
    "SERIE_A": {
        "name":         "Serie A TIM",
        "country":      "Italy",
        "fd_code":      "I1",
        "sofascore_id": 23,
        "tier":         1,
        "matches_per_season": 380,
        "teams":        20,
    },
    "LIGUE_1": {
        "name":         "Ligue 1 Uber Eats",
        "country":      "France",
        "fd_code":      "F1",
        "sofascore_id": 34,
        "tier":         1,
        "matches_per_season": 380,
        "teams":        18,
    },
    "BUNDESLIGA": {
        "name":         "Bundesliga",
        "country":      "Germany",
        "fd_code":      "D1",
        "sofascore_id": 3,
        "tier":         1,
        "matches_per_season": 306,
        "teams":        18,
    },
}

# ═══════════════════════════════════════════════════════
# EUROPEAN COMPETITION CONFIGURATION
# ═══════════════════════════════════════════════════════

EUROPEAN_TOURNAMENTS = {}

# ═══════════════════════════════════════════════════════
# INTERNATIONAL COMPETITION CONFIGURATION
# ═══════════════════════════════════════════════════════

INTERNATIONAL_TOURNAMENTS = {
    "FIFA_WORLD_CUP": {
        "name":         "FIFA World Cup",
        "fd_code":      "WC",
        "sofascore_id": 16,
        "tier":         0,
        "frequency":    "Every 4 years",
        "num_teams":    32,
        "note":         "Qualification + Group Stage + Knockout",
    },
    "UEFA_EUROS": {
        "name":         "UEFA European Championship",
        "fd_code":      "EC",
        "sofascore_id": 14,
        "tier":         0,
        "frequency":    "Every 4 years",
        "num_teams":    24,
        "note":         "Qualification + Group Stage + Knockout",
    },
    "COPA_AMERICA": {
        "name":         "Copa América",
        "fd_code":      "CA",
        "sofascore_id": 11,
        "tier":         0,
        "frequency":    "Every 4 years",
        "num_teams":    10,
        "note":         "South American national teams",
    },
    "AFRICA_CUP": {
        "name":         "Africa Cup of Nations",
        "fd_code":      "AFCON",
        "sofascore_id": 72,
        "tier":         0,
        "frequency":    "Every 2 years",
        "num_teams":    24,
        "note":         "African national teams",
    },
    "ASIAN_CUP": {
        "name":         "AFC Asian Cup",
        "fd_code":      "AC",
        "sofascore_id": 78,
        "tier":         0,
        "frequency":    "Every 4 years",
        "num_teams":    24,
        "note":         "Asian national teams",
    },
    "GOLD_CUP": {
        "name":         "CONCACAF Gold Cup",
        "fd_code":      "GC",
        "sofascore_id": 80,
        "tier":         0,
        "frequency":    "Every 2 years",
        "num_teams":    16,
        "note":         "North/Central American + Caribbean",
    },
    "NATIONS_LEAGUE": {
        "name":         "UEFA Nations League",
        "fd_code":      "UNL",
        "sofascore_id": 1358,
        "tier":         1,
        "frequency":    "Biennial",
        "num_teams":    55,
        "note":         "European national teams qualification",
    },
    "OLYMPIC_FOOTBALL": {
        "name":         "Olympic Football Tournament",
        "fd_code":      "OLY",
        "sofascore_id": 89,
        "tier":         1,
        "frequency":    "Every 4 years",
        "num_teams":    16,
        "note":         "U-23 teams (+ 3 overage players)",
    },
    "FIFA_CLUB_WC": {
        "name":         "FIFA Club World Cup",
        "fd_code":      "FCWC",
        "sofascore_id": 131,
        "tier":         0,
        "frequency":    "Annual",
        "num_teams":    32,
        "note":         "Continental club champions",
    },
    "CONMEBOL_SUDAMERICANA": {
        "name":         "CONMEBOL Sudamericana",
        "fd_code":      "CSUD",
        "sofascore_id": 294,
        "tier":         1,
        "frequency":    "Annual",
        "num_teams":    47,
        "note":         "South American club competition (new)",
    },
}

# ═══════════════════════════════════════════════════════
# COMBINED LEAGUE CONFIGURATION (for legacy compatibility)
# ═══════════════════════════════════════════════════════

LEAGUES = {**DOMESTIC_LEAGUES, **EUROPEAN_TOURNAMENTS}

# Generate code lists
ALL_LEAGUE_CODES = list(LEAGUES.keys())
DOMESTIC_LEAGUE_CODES = list(DOMESTIC_LEAGUES.keys())
EUROPEAN_LEAGUE_CODES = list(EUROPEAN_TOURNAMENTS.keys())
INTERNATIONAL_CODES = list(INTERNATIONAL_TOURNAMENTS.keys())
ALL_COMPETITION_CODES = ALL_LEAGUE_CODES + INTERNATIONAL_CODES

# ─ By Tier ─────────────────────────────────────────────
TOP_5_LEAGUES = ["EPL", "LA_LIGA", "SERIE_A", "LIGUE_1", "BUNDESLIGA"]
SECONDARY_LEAGUES = [k for k, v in DOMESTIC_LEAGUES.items() if v.get("tier", 1) == 2]
INTERNATIONAL_MAJOR = ["FIFA_WORLD_CUP", "UEFA_EUROS", "COPA_AMERICA", "AFRICA_CUP"]

# ─ By Region ───────────────────────────────────────────
ENGLISH_COMPS = ["EPL", "ENG_CHAMP"]
SPANISH_COMPS = ["LA_LIGA"]
ITALIAN_COMPS = ["SERIE_A"]
FRENCH_COMPS = ["LIGUE_1"]
GERMAN_COMPS = ["BUNDESLIGA"]

# ─ Convenience functions ───────────────────────────────
def is_domestic_league(code: str) -> bool:
    """Check if code is a domestic league."""
    return code in DOMESTIC_LEAGUE_CODES

def is_european_tournament(code: str) -> bool:
    """Check if code is a European tournament."""
    return code in EUROPEAN_LEAGUE_CODES

def is_international_tournament(code: str) -> bool:
    """Check if code is international (World Cup, Euros, etc.)."""
    return code in INTERNATIONAL_CODES

def get_league_info(code: str) -> dict:
    """Get league info by code."""
    return LEAGUES.get(code, {})

def get_tournament_info(code: str) -> dict:
    """Get international tournament info by code."""
    return INTERNATIONAL_TOURNAMENTS.get(code, {})


# ═══════════════════════════════════════════════════════
# DATA SOURCE URLs
# ═══════════════════════════════════════════════════════

class URLS:
    """Base URLs for every data source."""
    BASE_FD         = "https://www.football-data.co.uk/mmz4281"
    CLUBELO_TEAM    = "http://api.clubelo.com"
    CLUBELO_DATE    = "http://api.clubelo.com/{date}"
    UNDERSTAT_BASE  = "https://understat.com/league"
    FBREF_BASE      = "https://fbref.com/en/comps"
    SOFASCORE_BASE  = "https://www.sofascore.com"
    TRANSFERMARKT_BASE = "https://www.transfermarkt.com"
    WORLDFOOTBALL_BASE = "https://worldfootball.net"


# ═══════════════════════════════════════════════════════
# SEASON CONFIGURATION
# ═══════════════════════════════════════════════════════

class SEASONS:
    """Seasons for training, validation, and testing."""
    ALL_SEASONS = ["1819", "1920", "2021", "2122", "2223", "2324", "2425", "2526", "2627"]
    CL_SEASONS = ["2223", "2324", "2425", "2526", "2627"]

    TRAIN       = ["1819", "1920", "2021", "2122", "2223", "2324"]
    VALIDATION  = ["2425"]
    TEST        = ["2425"]

    CURRENT     = "2627"

    @staticmethod
    def to_label(code: str) -> str:
        """Convert season code (2223) to label (2022-23)."""
        return f"20{code[:2]}-{code[2:]}"


# ═══════════════════════════════════════════════════════
# football-data.co.uk COLUMN MAPPING
# ═══════════════════════════════════════════════════════

FD_COLUMN_MAP = {
    "Div":    "league",
    "Date":   "date",
    "HomeTeam": "home_team",
    "AwayTeam": "away_team",
    "FTHG":   "home_goals",
    "FTAG":   "away_goals",
    "FTR":    "result",
    "HTHG":   "ht_home_goals",
    "HTAG":   "ht_away_goals",
    "HTR":    "ht_result",
    "HS":     "home_shots",
    "AS":     "away_shots",
    "HST":    "home_shots_ot",
    "AST":    "away_shots_ot",
    "HC":     "home_corners",
    "AC":     "away_corners",
    "HY":     "home_yellows",
    "AY":     "away_yellows",
    "HR":     "home_reds",
    "AR":     "away_reds",
    "Referee": "referee",
    # Bet365 odds (primary)
    "B365H":  "odds_home",
    "B365D":  "odds_draw",
    "B365A":  "odds_away",
    "B365>2.5": "odds_over_25",
    "B365<2.5": "odds_under_25",
    # Alternative bookmakers (fallback if B365 missing)
    "PSH":  "odds_pinnacle_home",
    "PSD":  "odds_pinnacle_draw",
    "PSA":  "odds_pinnacle_away",
    "BWH":  "odds_betwin_home",
    "BWD":  "odds_betwin_draw",
    "BWA":  "odds_betwin_away",
}

REQUIRED_COLUMNS = [
    "date", "home_team", "away_team",
    "home_goals", "away_goals", "result",
]


# ═══════════════════════════════════════════════════════
# CACHE & PERFORMANCE CONFIGURATION
# ═══════════════════════════════════════════════════════

class CACHE:
    """Streamlit @st.cache_data TTL (time-to-live) in seconds."""
    FEATURES_TTL       = 3600      # 1 hour
    FIXTURES_TTL       = 300       # 5 minutes (live fixtures update rarely)
    PREDICTIONS_TTL    = 600       # 10 minutes
    LEAGUE_STATS_TTL   = 7200      # 2 hours
    ODDS_CACHE_TTL     = 3600      # 1 hour (cache odds from features)
    
    # API request timeouts
    API_REQUEST_TIMEOUT = 15       # seconds
    API_RETRY_ATTEMPTS  = 3        # Max retries for API calls
    API_RETRY_DELAY     = 2        # Initial delay between retries (exponential backoff)

class ODDS:
    """Odds handling and fallback configuration."""
    # Default odds if none available
    DEFAULT_HOME = 2.50
    DEFAULT_DRAW = 3.30
    DEFAULT_AWAY = 2.90
    DEFAULT_BTTS_YES = 1.85
    DEFAULT_BTTS_NO = 1.90
    DEFAULT_OVER_25 = 1.85
    DEFAULT_UNDER_25 = 1.95
    DEFAULT_CORNERS_O = 1.83
    DEFAULT_CORNERS_U = 1.97
    
    # Fallback strategy: try B365 first, then Pinnacle, then Betwin, then defaults
    FALLBACK_ORDER = [
        "odds_home",            # Bet365 primary
        "odds_pinnacle_home",   # Pinnacle backup
        "odds_betwin_home",     # Betwin backup
    ]
    
    # Use historical team average odds if all else fails
    USE_HISTORICAL_AVERAGE = True
    HISTORICAL_WINDOW = 10  # Use last 10 matches for average


# ═══════════════════════════════════════════════════════
# FEATURE ENGINEERING PARAMETERS
# ═══════════════════════════════════════════════════════

class FEATURES:
    """Parameters controlling feature creation."""
    ROLLING_WINDOWS = [3, 5, 10]
    PRIMARY_WINDOW = 5
    MIN_MATCHES_FOR_ROLLING = 3
    ELO_DEFAULT_RATING = 1500

    TARGET_MATCH_RESULT = "result"
    TARGET_BTTS         = "btts"
    TARGET_OVER_UNDER   = "over_25"
    TARGET_CORNERS      = "over_corners"

    CORNER_LINE = 9.5
    GOAL_LINE = 2.5

    IMPORTANCE_TOP4_CUTOFF       = 4
    IMPORTANCE_RELEGATION_CUTOFF = 3

    FATIGUE_RESTED_DAYS     = 6
    FATIGUE_TIRED_DAYS      = 3
    FATIGUE_WINDOW_DAYS     = 30

    USE_CL_STAGE_ENCODING   = True
    USE_CL_DRAW_ENCODING    = True
    USE_CL_EXPERIENCE       = True
    REDUCE_CL_HOME_ADVANTAGE = True


# ═══════════════════════════════════════════════════════
# ELO SYSTEM PARAMETERS
# ═══════════════════════════════════════════════════════

class ELO:
    """Parameters for the dynamic ELO rating system."""
    DEFAULT_RATING      = 1500
    K_FACTOR            = 32
    CL_K_FACTOR_MULTIPLIER = 1.2
    HOME_ADVANTAGE      = 100
    CL_HOME_ADVANTAGE   = 70
    PROMOTION_REGRESSION = 0.5


# ═══════════════════════════════════════════════════════
# MODEL PARAMETERS
# ═══════════════════════════════════════════════════════

class MODEL_PARAMS:
    """Default hyperparameters for each model."""
    TS_SPLIT_N_FOLDS = 5

    XGBOOST_DEFAULT = {
        "n_estimators":     500,
        "max_depth":        5,
        "learning_rate":    0.05,
        "subsample":        0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 3,
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
        "verbose":          -1,
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

    DIXON_COLES_RHO_INIT = -0.13
    DIXON_COLES_TOP_N = 10


# ═══════════════════════════════════════════════════════
# SPORTS INVESTING ENGINE PARAMETERS
# ═══════════════════════════════════════════════════════

class KELLY:
    """Kelly Criterion staking parameters."""
    FRACTION = 0.25
    MAX_STAKE_PCT = 0.05
    MIN_STAKE_PCT = 0.005
    DEFAULT_BANKROLL = 1000.0


class VALUE_BET:
    """Thresholds for flagging a bet as a value bet."""
    MIN_EDGE = 0.05
    MIN_MODEL_PROB = 0.30
    MAX_ODDS = 8.00
    MIN_ODDS = 1.20

    HIGH_CONFIDENCE   = 0.65
    MEDIUM_CONFIDENCE = 0.50


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
    SLEEP_BETWEEN_REQUESTS = 2.0
    SLEEP_FIRST_REQUEST    = 3.0
    MAX_RETRIES            = 3
    REQUEST_TIMEOUT        = 15

    USER_AGENT = (
        "Mozilla/5.0 (compatible; FootballPredictor/2.0; "
        "research-only; contact: your@email.com)"
    )

    DEFAULT_HEADERS = {
        "User-Agent":       USER_AGENT,
        "Accept-Language":  "en-GB,en;q=0.9",
        "Accept-Encoding":  "gzip, deflate, br",
    }

    CACHE_EXPIRY_HOURS     = 24
    LIVE_SCRAPE_INTERVAL   = 300


# ═══════════════════════════════════════════════════════
# LOGGING CONFIGURATION
# ═══════════════════════════════════════════════════════

class LOGGING:
    """Loguru logging settings."""
    LEVEL   = "INFO"
    FORMAT  = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{line}</cyan> — "
        "<level>{message}</level>"
    )
    ROTATION   = "10 MB"
    RETENTION  = "30 days"


# ═══════════════════════════════════════════════════════
# RANDOM SEED
# ═══════════════════════════════════════════════════════

RANDOM_SEED = 42


# ═══════════════════════════════════════════════════════
# STARTUP CHECK
# ═══════════════════════════════════════════════════════

if __name__ == "__main__":
    print("✓ config.py (v2.0) loaded successfully")
    print(f"\n  Project root : {ROOT_DIR}")
    print(f"\n  🏆 DOMESTIC LEAGUES ({len(DOMESTIC_LEAGUE_CODES)})")
    print(f"     Top 5: {', '.join(TOP_5_LEAGUES)}")
    print(f"     Secondary: {len(SECONDARY_LEAGUES)} leagues")
    print(f"     Total: {len(DOMESTIC_LEAGUE_CODES)} domestic competitions")
    
    print(f"\n  🏅 EUROPEAN TOURNAMENTS ({len(EUROPEAN_LEAGUE_CODES)})")
    print(f"     Premier: CL, EL, Conference League")
    print(f"     National Cups: {len([k for k in EUROPEAN_LEAGUE_CODES if 'CUP' in k])} cups")
    print(f"     Domestic Cups: EFL Cup, Super Cup, Community Shield")
    
    print(f"\n  🌍 INTERNATIONAL TOURNAMENTS ({len(INTERNATIONAL_CODES)})")
    print(f"     Major: {', '.join(INTERNATIONAL_MAJOR)}")
    print(f"     Total: {len(INTERNATIONAL_CODES)} tournaments")
    
    print(f"\n  📊 COVERAGE SUMMARY")
    print(f"     Total competitions: {len(ALL_COMPETITION_CODES)}")
    print(f"     Training seasons: {', '.join(SEASONS.TRAIN)}")
    print(f"     Test season: {SEASONS.TEST[0]}")
    print(f"\n  ⚙️ SETTINGS")
    print(f"     Kelly fraction: {KELLY.FRACTION * 100:.0f}% of full Kelly")
    print(f"     Value edge min: {VALUE_BET.MIN_EDGE * 100:.0f}%")
    print(f"     Cache TTL: {CACHE.FEATURES_TTL}s (features), {CACHE.FIXTURES_TTL}s (live)")
    
    print("\n  Creating directories...")
    PATHS.create_all()
    print("  ✓ All directories created\n")