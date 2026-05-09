"""
update_data.py — The Oracle of the Pitch: Full Data Pipeline (API-only)
========================================================================
Downloads latest data exclusively from football-data.org API,
engineers rich features, retrains model, predicts today.

Run:
  python src/update_data.py                  # full update → features → train → predict
  python src/update_data.py --no-train       # refresh data only
  python src/update_data.py --predict-only   # skip download, just predict today
  python src/update_data.py --league EPL     # single league
  python src/update_data.py --api-key XXXX   # override API key

Requirements:
  pip install requests pandas numpy xgboost scikit-learn joblib loguru pyarrow
"""

import sys, re, os, time, datetime, warnings, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
warnings.filterwarnings("ignore")

import requests
import pandas as pd
import numpy as np
import joblib
from loguru import logger

# ── Try importing config; fall back to sensible defaults ──────────────────────
try:
    from config import PATHS, LEAGUES, SEASONS, KELLY, REQUIRED_COLUMNS
except ImportError:
    from types import SimpleNamespace

    class _Paths:
        BASE      = Path(__file__).parent.parent
        DATA      = BASE / "data"
        RAW       = DATA / "raw"
        PROCESSED = DATA / "processed"
        MODELS    = BASE / "models"
        MODEL_MATCH_RESULT = MODELS / "match_result.pkl"
        def create_all(self):
            for p in [self.RAW, self.PROCESSED, self.MODELS]:
                p.mkdir(parents=True, exist_ok=True)

    PATHS = _Paths()

    LEAGUES = {
        "EPL":       {"name": "Premier League", "country": "England", "fdorg_code": "PL"},
        "LA_LIGA":   {"name": "La Liga",        "country": "Spain",   "fdorg_code": "PD"},
        "SERIE_A":   {"name": "Serie A",        "country": "Italy",   "fdorg_code": "SA"},
        "BUNDESLIGA":{"name": "Bundesliga",      "country": "Germany", "fdorg_code": "BL1"},
        "LIGUE_1":   {"name": "Ligue 1",        "country": "France",  "fdorg_code": "FL1"},
        "ENG_CHAMP": {"name": "Championship",   "country": "England", "fdorg_code": "ELC"},
    }

    SEASONS = SimpleNamespace(
        ALL_SEASONS=["2021", "2022", "2023", "2024"],
        TRAIN=["2021", "2022", "2023"],
        TEST=["2024"],
    )

    KELLY = SimpleNamespace(FRACTION=0.25, MAX_STAKE_PCT=0.05)

    REQUIRED_COLUMNS = ["date", "home_team", "away_team", "result"]


logger.remove()
logger.add(sys.stderr, level="INFO", colorize=True,
           format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}")

TODAY        = datetime.date.today().isoformat()
CURRENT_YEAR = datetime.date.today().year

# ── Team name normalisation ────────────────────────────────────────────────────
TEAM_NAME_MAP = {
    # EPL
    "Arsenal FC": "Arsenal", "Chelsea FC": "Chelsea", "Liverpool FC": "Liverpool",
    "Manchester City FC": "Man City", "Manchester United FC": "Man United",
    "Tottenham Hotspur FC": "Tottenham", "Newcastle United FC": "Newcastle",
    "Aston Villa FC": "Aston Villa", "West Ham United FC": "West Ham",
    "Brighton & Hove Albion FC": "Brighton", "Brentford FC": "Brentford",
    "Fulham FC": "Fulham", "Crystal Palace FC": "Crystal Palace",
    "Wolverhampton Wanderers FC": "Wolves", "Everton FC": "Everton",
    "Nottingham Forest FC": "Nott'm Forest", "AFC Bournemouth": "Bournemouth",
    "Leicester City FC": "Leicester", "Ipswich Town FC": "Ipswich",
    "Southampton FC": "Southampton", "Leeds United FC": "Leeds",
    "Burnley FC": "Burnley", "Sheffield United FC": "Sheffield United",
    "Luton Town FC": "Luton",
    # La Liga
    "FC Barcelona": "Barcelona", "Real Madrid CF": "Real Madrid",
    "Club Atlético de Madrid": "Ath Madrid", "Athletic Club": "Ath Bilbao",
    "Real Sociedad de Fútbol": "Sociedad", "Villarreal CF": "Villarreal",
    "Sevilla FC": "Sevilla", "Real Betis Balompié": "Betis",
    "Rayo Vallecano de Madrid": "Vallecano", "Getafe CF": "Getafe",
    "RC Celta de Vigo": "Celta", "CA Osasuna": "Osasuna",
    "Deportivo Alavés": "Alaves", "Valencia CF": "Valencia",
    "UD Las Palmas": "Las Palmas", "RCD Mallorca": "Mallorca",
    "Girona FC": "Girona", "Real Valladolid CF": "Valladolid",
    "CD Leganés": "Leganes", "Espanyol de Barcelona": "Espanyol",
    "Cádiz CF": "Cadiz", "UD Almería": "Almeria",
    "Real Granada CF": "Granada", "Levante UD": "Levante",
    # Serie A
    "Inter Milan": "Inter", "AC Milan": "Milan", "Juventus FC": "Juventus",
    "SSC Napoli": "Napoli", "AS Roma": "Roma", "SS Lazio": "Lazio",
    "ACF Fiorentina": "Fiorentina", "Atalanta BC": "Atalanta",
    "Bologna FC 1909": "Bologna", "Torino FC": "Torino", "Udinese Calcio": "Udinese",
    "Cagliari Calcio": "Cagliari", "Genoa CFC": "Genoa", "US Lecce": "Lecce",
    "Hellas Verona FC": "Verona", "Empoli FC": "Empoli", "Venezia FC": "Venezia",
    "Parma Calcio 1913": "Parma", "Como 1907": "Como", "AC Monza": "Monza",
    "US Salernitana 1919": "Salernitana", "US Frosinone Calcio": "Frosinone",
    "US Sassuolo Calcio": "Sassuolo",
    # Bundesliga
    "FC Bayern München": "Bayern Munich", "Borussia Dortmund": "Dortmund",
    "Bayer 04 Leverkusen": "Leverkusen", "RB Leipzig": "RB Leipzig",
    "Eintracht Frankfurt": "Ein Frankfurt", "SC Freiburg": "Freiburg",
    "VfB Stuttgart": "Stuttgart", "1. FC Union Berlin": "Union Berlin",
    "1. FSV Mainz 05": "Mainz", "Borussia Mönchengladbach": "M'gladbach",
    "TSG 1899 Hoffenheim": "Hoffenheim", "FC Augsburg": "Augsburg",
    "Werder Bremen": "Werder Bremen", "VfL Wolfsburg": "Wolfsburg",
    "VfL Bochum 1848": "Bochum", "1. FC Heidenheim 1846": "Heidenheim",
    "FC St. Pauli 1910": "St Pauli", "Holstein Kiel": "Holstein Kiel",
    "FC Schalke 04": "Schalke", "Hertha BSC": "Hertha",
    # Ligue 1
    "Paris Saint-Germain FC": "Paris SG", "Olympique de Marseille": "Marseille",
    "Olympique Lyonnais": "Lyon", "AS Monaco FC": "Monaco", "LOSC Lille": "Lille",
    "RC Strasbourg Alsace": "Strasbourg", "OGC Nice": "Nice",
    "Stade Rennais FC 1901": "Rennes", "RC Lens": "Lens",
    "Stade Brestois 29": "Brest", "Montpellier HSC": "Montpellier",
    "FC Nantes": "Nantes", "Toulouse FC": "Toulouse", "Le Havre AC": "Le Havre",
    "AJ Auxerre": "Auxerre", "Angers SCO": "Angers",
    "AS Saint-Étienne": "St Etienne", "Stade de Reims": "Reims",
    "Racing Club de Lens": "Lens", "FC Lorient": "Lorient",
    "Clermont Foot 63": "Clermont", "AC Ajaccio": "Ajaccio",
    # Championship
    "Hull City AFC": "Hull", "Millwall FC": "Millwall",
    "Queens Park Rangers FC": "QPR", "Coventry City FC": "Coventry",
    "Cardiff City FC": "Cardiff", "Swansea City AFC": "Swansea",
    "Middlesbrough FC": "Middlesbrough", "Stoke City FC": "Stoke",
    "Watford FC": "Watford", "West Bromwich Albion FC": "West Brom",
    "Norwich City FC": "Norwich", "Bristol City FC": "Bristol City",
    "Preston North End FC": "Preston", "Sheffield Wednesday FC": "Sheffield Weds",
    "Sunderland AFC": "Sunderland", "Birmingham City FC": "Birmingham",
    "Blackburn Rovers FC": "Blackburn", "Plymouth Argyle FC": "Plymouth",
    "Rotherham United FC": "Rotherham", "Huddersfield Town AFC": "Huddersfield",
}


# ════════════════════════════════════════════════════════════════════
# API KEY
# ════════════════════════════════════════════════════════════════════

def load_api_key(provided=None):
    if provided:
        return provided
    key = os.environ.get("FOOTBALL_DATA_KEY", "").strip()
    if key:
        return key
    for env_file in [Path(".env"), Path(__file__).parent.parent / ".env"]:
        if env_file.exists():
            for line in env_file.read_text().splitlines():
                if line.startswith("FOOTBALL_DATA_KEY="):
                    return line.split("=", 1)[1].strip().strip("\"'")
    return None


# ════════════════════════════════════════════════════════════════════
# STEP 1 — FETCH RESULTS FROM API
# ════════════════════════════════════════════════════════════════════

def _api_get(url, headers, retries=3, backoff=65):
    """GET with automatic rate-limit retry."""
    for attempt in range(retries):
        try:
            r = requests.get(url, headers=headers, timeout=15)
            if r.status_code == 200:
                return r
            if r.status_code == 429:
                wait = backoff * (attempt + 1)
                logger.warning(f"   Rate limited — waiting {wait}s ...")
                time.sleep(wait)
                continue
            if r.status_code == 403:
                logger.error("   API key invalid or expired.")
                return None
            logger.warning(f"   HTTP {r.status_code} for {url}")
            return None
        except Exception as e:
            logger.warning(f"   Request error ({attempt+1}/{retries}): {e}")
            time.sleep(5)
    return None


def fetch_all_results(api_key, league_key=None):
    """
    Fetch ALL finished matches from football-data.org for every
    configured league, seasons 2021 → present.
    Returns a tidy DataFrame.
    """
    if not api_key:
        raise ValueError(
            "A football-data.org API key is required.\n"
            "Get a free key at https://www.football-data.org/client/register\n"
            "Then add to .env:  FOOTBALL_DATA_KEY=your_key_here"
        )

    logger.info("🌐 Fetching all results from football-data.org API ...")
    headers = {"X-Auth-Token": api_key}
    frames  = []

    leagues = (
        {league_key: LEAGUES[league_key]}
        if league_key and league_key in LEAGUES
        else LEAGUES
    )

    for lk, info in leagues.items():
        code = info.get("fdorg_code")
        if not code:
            continue

        league_rows = 0
        for season_year in range(2021, CURRENT_YEAR + 1):
            url = (
                f"https://api.football-data.org/v4/competitions/{code}/matches"
                f"?season={season_year}&status=FINISHED"
            )
            time.sleep(0.7)
            r = _api_get(url, headers)
            if r is None:
                continue

            for m in r.json().get("matches", []):
                sc = m.get("score", {}).get("fullTime", {})
                hg, ag = sc.get("home"), sc.get("away")
                if hg is None or ag is None:
                    continue

                ht_raw = m["homeTeam"]["name"]
                at_raw = m["awayTeam"]["name"]
                ht  = TEAM_NAME_MAP.get(ht_raw, ht_raw)
                at  = TEAM_NAME_MAP.get(at_raw, at_raw)
                res = "H" if hg > ag else "A" if ag > hg else "D"

                ht_score = m.get("score", {}).get("halfTime", {})
                ht_hg = ht_score.get("home")
                ht_ag = ht_score.get("away")

                frames.append({
                    "date":          pd.to_datetime(m["utcDate"]).normalize(),
                    "home_team":     ht,
                    "away_team":     at,
                    "home_goals":    int(hg),
                    "away_goals":    int(ag),
                    "ht_home_goals": int(ht_hg) if ht_hg is not None else np.nan,
                    "ht_away_goals": int(ht_ag) if ht_ag is not None else np.nan,
                    "result":        res,
                    "league_key":    lk,
                    "season":        str(season_year),
                    "matchday":      m.get("matchday", np.nan),
                    "source":        "fdorg",
                })
                league_rows += 1

        logger.info(f"   ✓ {lk:<12} {league_rows:>5} matches fetched")

    if not frames:
        raise RuntimeError(
            "No data returned from API. "
            "Check your API key and network connection."
        )

    df = pd.DataFrame(frames)
    df["date"] = pd.to_datetime(df["date"])
    df.sort_values(["league_key", "date"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


# ════════════════════════════════════════════════════════════════════
# STEP 2 — BUILD MASTER PARQUET
# ════════════════════════════════════════════════════════════════════

def add_base_targets(df):
    df["total_goals"]    = df["home_goals"] + df["away_goals"]
    df["goal_diff"]      = df["home_goals"] - df["away_goals"]
    df["btts"]           = ((df["home_goals"] >= 1) & (df["away_goals"] >= 1)).astype(int)
    df["over_25"]        = (df["total_goals"] > 2.5).astype(int)
    df["over_35"]        = (df["total_goals"] > 3.5).astype(int)
    df["result_numeric"] = df["result"].map({"H": 0, "D": 1, "A": 2})
    return df


def build_master(api_df):
    """Deduplicate, add targets, save master.parquet."""
    logger.info("🔨 Building master parquet ...")

    df = api_df.copy()
    df.drop_duplicates(
        subset=["date", "home_team", "away_team", "league_key"], inplace=True
    )
    df = df[df["result"].isin(["H", "D", "A"])].copy()
    df.dropna(subset=REQUIRED_COLUMNS, inplace=True)

    df = add_base_targets(df)

    df["season_label"] = df["season"].apply(
        lambda s: f"{s}/{str(int(s)+1)[2:]}" if str(s).isdigit() else str(s)
    )
    df["league_name"] = df["league_key"].map(
        {k: v["name"] for k, v in LEAGUES.items()}
    )
    df["country"] = df["league_key"].map(
        {k: v.get("country", "") for k, v in LEAGUES.items()}
    )

    PATHS.PROCESSED.mkdir(parents=True, exist_ok=True)
    out = PATHS.PROCESSED / "master.parquet"
    df.to_parquet(out, index=False, engine="pyarrow")
    logger.info(
        f"   ✓ Master saved: {len(df):,} matches | "
        f"latest: {df['date'].max().date()}"
    )
    return df


# ════════════════════════════════════════════════════════════════════
# STEP 3 — FEATURE ENGINEERING
# ════════════════════════════════════════════════════════════════════

def compute_elo(df, k=32.0, initial=1500.0):
    """Compute per-league Elo, assigning pre-match values to each row."""
    logger.info("📐 Computing Elo ratings ...")
    df = df.sort_values(["league_key", "date"]).copy()
    ratings = {}
    home_elo_pre, away_elo_pre   = [], []
    home_elo_post, away_elo_post = [], []

    for _, row in df.iterrows():
        lk = row["league_key"]
        if lk not in ratings:
            ratings[lk] = {}
        ht, at = row["home_team"], row["away_team"]
        he = ratings[lk].get(ht, initial)
        ae = ratings[lk].get(at, initial)
        home_elo_pre.append(he)
        away_elo_pre.append(ae)

        adj_exp_h = 1 / (1 + 10 ** ((ae - he - 100) / 400))
        result = row.get("result", "")
        if result == "H":   s_h, s_a = 1.0, 0.0
        elif result == "A": s_h, s_a = 0.0, 1.0
        else:               s_h, s_a = 0.5, 0.5

        ratings[lk][ht] = he + k * (s_h - adj_exp_h)
        ratings[lk][at] = ae + k * (s_a - (1 - adj_exp_h))
        home_elo_post.append(ratings[lk][ht])
        away_elo_post.append(ratings[lk][at])

    df["home_elo"]       = home_elo_pre
    df["away_elo"]       = away_elo_pre
    df["home_elo_after"] = home_elo_post
    df["away_elo_after"] = away_elo_post
    df["elo_diff"]       = df["home_elo"] - df["away_elo"]
    return df


def _col_mean(df, col):
    """
    Safely return the mean of a column, or NaN if absent.
    Guards against duplicate-column DataFrames after pd.concat.
    """
    if col not in df.columns:
        return np.nan
    series = df[col]
    if isinstance(series, pd.DataFrame):
        series = series.iloc[:, 0]
    return series.mean()


def _rolling_team_stats(df, team, as_home, before_date, n=5):
    """Rolling n-game stats for a team before a given date."""
    mask = (
        (df["home_team"] == team) if as_home else (df["away_team"] == team)
    ) & (df["date"] < before_date)
    recent = df[mask].sort_values("date").tail(n)

    if recent.empty:
        return {}

    if as_home:
        gf    = _col_mean(recent, "home_goals")
        ga    = _col_mean(recent, "away_goals")
        wins  = (recent["result"] == "H").sum()
        draws = (recent["result"] == "D").sum()
    else:
        gf    = _col_mean(recent, "away_goals")
        ga    = _col_mean(recent, "home_goals")
        wins  = (recent["result"] == "A").sum()
        draws = (recent["result"] == "D").sum()

    losses = n - wins - draws
    pts    = wins * 3 + draws
    gd     = (gf - ga) if pd.notna(gf) and pd.notna(ga) else np.nan
    return {
        "gf": gf, "ga": ga, "gd": gd,
        "wins": wins, "draws": draws, "losses": losses, "pts": pts,
        "form": pts / (n * 3),
    }


def engineer_features(df, n_form=5):
    """Build the rich feature set used for training and prediction."""
    logger.info(f"⚙️  Engineering features (rolling n={n_form}) ...")

    df = compute_elo(df)
    df = df.sort_values(["league_key", "date"]).reset_index(drop=True)

    records = []

    for idx, row in df.iterrows():
        lk   = row["league_key"]
        sub  = df[df["league_key"] == lk]
        date = row["date"]
        ht   = row["home_team"]
        at   = row["away_team"]

        feat = {}

        # ── Venue-specific rolling form ────────────────────────────
        hs  = _rolling_team_stats(sub, ht, as_home=True,  before_date=date, n=n_form)
        as_ = _rolling_team_stats(sub, at, as_home=False, before_date=date, n=n_form)
        for k, v in hs.items():  feat[f"home_{k}_home"] = v
        for k, v in as_.items(): feat[f"away_{k}_away"] = v

        # ── Overall form (any venue, last n) ───────────────────────
        h_all = pd.concat([
            sub[(sub["home_team"] == ht) & (sub["date"] < date)],
            sub[(sub["away_team"] == ht) & (sub["date"] < date)],
        ]).sort_values("date").tail(n_form)
        a_all = pd.concat([
            sub[(sub["home_team"] == at) & (sub["date"] < date)],
            sub[(sub["away_team"] == at) & (sub["date"] < date)],
        ]).sort_values("date").tail(n_form)

        feat["home_overall_form"] = (
            (
                ((h_all["result"] == "H") & (h_all["home_team"] == ht)).sum() +
                ((h_all["result"] == "A") & (h_all["away_team"] == ht)).sum()
            ) / max(1, len(h_all))
        ) if not h_all.empty else 0.5

        feat["away_overall_form"] = (
            (
                ((a_all["result"] == "A") & (a_all["away_team"] == at)).sum() +
                ((a_all["result"] == "H") & (a_all["home_team"] == at)).sum()
            ) / max(1, len(a_all))
        ) if not a_all.empty else 0.5

        # ── Elo ────────────────────────────────────────────────────
        feat["home_elo"] = row.get("home_elo", 1500)
        feat["away_elo"] = row.get("away_elo", 1500)
        feat["elo_diff"] = row.get("elo_diff", 0)

        # ── Head-to-head (last 5) ──────────────────────────────────
        h2h = sub[
            ((sub["home_team"] == ht) & (sub["away_team"] == at)) |
            ((sub["home_team"] == at) & (sub["away_team"] == ht))
        ]
        h2h = h2h[h2h["date"] < date].sort_values("date").tail(5)
        if not h2h.empty:
            n_h2h  = len(h2h)
            h_wins = (
                ((h2h["home_team"] == ht) & (h2h["result"] == "H")).sum() +
                ((h2h["away_team"] == ht) & (h2h["result"] == "A")).sum()
            )
            a_wins = (
                ((h2h["home_team"] == at) & (h2h["result"] == "H")).sum() +
                ((h2h["away_team"] == at) & (h2h["result"] == "A")).sum()
            )
            draws = (h2h["result"] == "D").sum()
            feat["h2h_home_win_rate"] = h_wins / n_h2h
            feat["h2h_away_win_rate"] = a_wins / n_h2h
            feat["h2h_draw_rate"]     = draws  / n_h2h
            feat["h2h_avg_goals"]     = (
                _col_mean(h2h, "total_goals")
                if "total_goals" in h2h.columns else 2.5
            )
            feat["h2h_count"] = n_h2h
        else:
            feat["h2h_home_win_rate"] = 0.45
            feat["h2h_away_win_rate"] = 0.28
            feat["h2h_draw_rate"]     = 0.27
            feat["h2h_avg_goals"]     = 2.5
            feat["h2h_count"]         = 0

        # ── Season-level home/away dominance ───────────────────────
        season_home = sub[(sub["home_team"] == ht) & (sub["date"] < date)]
        if len(season_home) >= 3:
            feat["home_team_home_win_rate_season"] = (
                season_home["result"] == "H"
            ).mean()
            feat["home_team_avg_gf_season"] = _col_mean(season_home, "home_goals")
            feat["home_team_avg_ga_season"] = _col_mean(season_home, "away_goals")
        else:
            feat["home_team_home_win_rate_season"] = 0.45
            feat["home_team_avg_gf_season"]        = 1.5
            feat["home_team_avg_ga_season"]        = 1.2

        season_away = sub[(sub["away_team"] == at) & (sub["date"] < date)]
        if len(season_away) >= 3:
            feat["away_team_away_win_rate_season"] = (
                season_away["result"] == "A"
            ).mean()
            feat["away_team_avg_gf_season"] = _col_mean(season_away, "away_goals")
            feat["away_team_avg_ga_season"] = _col_mean(season_away, "home_goals")
        else:
            feat["away_team_away_win_rate_season"] = 0.28
            feat["away_team_avg_gf_season"]        = 1.2
            feat["away_team_avg_ga_season"]        = 1.5

        # ── Goal expectation proxy ─────────────────────────────────
        feat["expected_home_goals"]  = (
            feat.get("home_gf_home", 1.5) + feat.get("away_ga_away", 1.5)
        ) / 2
        feat["expected_away_goals"]  = (
            feat.get("away_gf_away", 1.2) + feat.get("home_ga_home", 1.2)
        ) / 2
        feat["expected_total_goals"] = (
            feat["expected_home_goals"] + feat["expected_away_goals"]
        )
        feat["expected_goal_diff"] = (
            feat["expected_home_goals"] - feat["expected_away_goals"]
        )

        records.append(feat)

    feat_df = pd.DataFrame(records)

    def _san(c):
        return re.sub(r'[^a-zA-Z0-9_]', '_', str(c))
    feat_df.columns = [_san(c) for c in feat_df.columns]

    meta_cols = [
        "date", "home_team", "away_team", "league_key", "season",
        "season_label", "league_name", "country", "result",
        "home_goals", "away_goals", "total_goals", "btts",
        "over_25", "over_35", "result_numeric",
    ]
    meta = df[[c for c in meta_cols if c in df.columns]].reset_index(drop=True)
    out  = pd.concat([meta, feat_df], axis=1)
    out.sort_values(["league_key", "date"], inplace=True)
    out.reset_index(drop=True, inplace=True)

    save_path = PATHS.PROCESSED / "features_master.parquet"
    out.to_parquet(save_path, index=False, engine="pyarrow")
    logger.info(f"   ✓ Features saved: {len(out):,} rows × {len(out.columns)} cols")
    return out


# ════════════════════════════════════════════════════════════════════
# STEP 4 — TRAIN ORACLE MODEL
# ════════════════════════════════════════════════════════════════════

def train_oracle(features_df):
    """
    Train XGBoost with temporal split to avoid leakage.
    Returns (model, feature_names).
    """
    try:
        import xgboost as xgb
        from sklearn.metrics import accuracy_score, f1_score
    except ImportError:
        raise ImportError("pip install xgboost scikit-learn")

    logger.info("🤖 Training Oracle model ...")

    df = features_df.copy()
    seen, nc = {}, {}
    for col in df.columns:
        c = re.sub(r'[^a-zA-Z0-9_]', '_', str(col))
        if c in seen: seen[c] += 1; c = f"{c}_{seen[c]}"
        else: seen[c] = 0
        nc[col] = c
    df = df.rename(columns=nc)

    exclude = {
        'result', 'btts', 'over_25', 'over_35', 'total_goals',
        'goal_diff', 'result_numeric', 'home_goals', 'away_goals',
        'ht_home_goals', 'ht_away_goals', 'home_elo_after', 'away_elo_after',
        'date', 'home_team', 'away_team', 'league_key', 'season',
        'season_label', 'league_name', 'country', 'source', 'matchday',
    }

    feature_cols = [
        c for c in df.columns
        if c not in exclude and pd.api.types.is_numeric_dtype(df[c])
    ]

    valid = df[df["result"].isin(["H", "D", "A"])].copy()
    X = (
        valid[feature_cols]
        .apply(pd.to_numeric, errors='coerce')
        .fillna(-999)
        .astype(np.float32)
    )
    y = valid["result"].map({"H": 0, "D": 1, "A": 2}).astype(np.int32)

    seasons = sorted(valid["season"].unique())
    if len(seasons) >= 3:
        train_mask = valid["season"] != seasons[-1]
        val_mask   = valid["season"] == seasons[-2]
        test_mask  = valid["season"] == seasons[-1]
    else:
        n     = len(X)
        split = int(n * 0.8)
        train_mask = pd.Series(
            [True]*split + [False]*(n - split), index=valid.index
        )
        val_mask  = ~train_mask
        test_mask = ~train_mask

    X_train, y_train = X[train_mask], y[train_mask]
    X_val,   y_val   = X[val_mask],   y[val_mask]
    X_test,  y_test  = X[test_mask],  y[test_mask]

    logger.info(
        f"   Train: {len(X_train):,}  Val: {len(X_val):,}  Test: {len(X_test):,}"
    )

    model = xgb.XGBClassifier(
        n_estimators=1000, max_depth=5, learning_rate=0.03,
        subsample=0.80, colsample_bytree=0.75, min_child_weight=5,
        gamma=0.1, reg_alpha=0.05, reg_lambda=1.0,
        objective="multi:softprob", eval_metric="mlogloss",
        num_class=3, random_state=42, n_jobs=-1, tree_method="hist",
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
        early_stopping_rounds=50,
    )

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    logger.info(f"   ✓ Test accuracy: {acc:.3f}  |  F1 (weighted): {f1:.3f}")
    logger.info(f"   ✓ Best iteration: {model.best_iteration}")

    PATHS.MODELS.mkdir(parents=True, exist_ok=True)
    joblib.dump(model,        PATHS.MODEL_MATCH_RESULT)
    joblib.dump(feature_cols, PATHS.MODELS / "match_result_features.pkl")
    joblib.dump({
        "accuracy": acc, "f1": f1,
        "n_train": len(X_train), "n_test": len(X_test),
        "feature_count": len(feature_cols),
        "trained_at": TODAY,
    }, PATHS.MODELS / "model_meta.pkl")

    logger.info(f"   ✓ Model saved → {PATHS.MODEL_MATCH_RESULT}")
    return model, feature_cols


# ════════════════════════════════════════════════════════════════════
# STEP 5 — FETCH TODAY'S FIXTURES + PREDICT
# ════════════════════════════════════════════════════════════════════

def fetch_todays_fixtures(api_key, league_key=None):
    """Fetch today's fixtures from the API."""
    if not api_key:
        logger.warning(
            "\n  ⚠  No API key — falling back to most recent matchday as demo.\n"
            "  Get a FREE key: https://www.football-data.org/client/register\n"
            "  Then add to .env:  FOOTBALL_DATA_KEY=your_key_here\n"
        )
        return _fallback_fixtures(league_key)

    leagues  = (
        {league_key: LEAGUES[league_key]}
        if league_key and league_key in LEAGUES
        else LEAGUES
    )
    headers  = {"X-Auth-Token": api_key}
    fixtures = []

    for lk, info in leagues.items():
        code = info.get("fdorg_code")
        if not code:
            continue
        url = (
            f"https://api.football-data.org/v4/competitions/{code}/matches"
            f"?dateFrom={TODAY}&dateTo={TODAY}"
        )
        time.sleep(0.7)
        r = _api_get(url, headers)
        if r is None:
            continue

        for m in r.json().get("matches", []):
            ht_raw = m["homeTeam"]["name"]
            at_raw = m["awayTeam"]["name"]
            ht = TEAM_NAME_MAP.get(ht_raw, ht_raw)
            at = TEAM_NAME_MAP.get(at_raw, at_raw)
            fixtures.append({
                "home_team":  ht,
                "away_team":  at,
                "league_key": lk,
                "match_date": TODAY,
                "kickoff":    m.get("utcDate", "")[:16],
                "status":     m.get("status", "SCHEDULED"),
                "matchday":   m.get("matchday", ""),
                "source":     "live",
            })
        n = sum(1 for f in fixtures if f["league_key"] == lk)
        logger.info(f"   {lk}: {n} fixtures today")

    if not fixtures:
        logger.info("No live fixtures today — using most recent matchday as demo.")
        return _fallback_fixtures(league_key)

    return fixtures


def _fallback_fixtures(league_key=None):
    """Return the most recent matchday from stored data as demo fixtures."""
    for path in [
        PATHS.PROCESSED / "features_master.parquet",
        PATHS.PROCESSED / "master.parquet",
    ]:
        if path.exists():
            break
    else:
        logger.error("No processed data found. Run the full pipeline first.")
        return []

    df = pd.read_parquet(path, engine="pyarrow")
    df["date"] = pd.to_datetime(df["date"])
    if league_key:
        df = df[df["league_key"] == league_key]

    fixtures = []
    for lk in df["league_key"].unique():
        sub    = df[df["league_key"] == lk]
        latest = sub["date"].max()
        for _, row in sub[sub["date"] == latest].iterrows():
            fixtures.append({
                "home_team":  row["home_team"],
                "away_team":  row["away_team"],
                "league_key": lk,
                "match_date": str(latest.date()),
                "kickoff":    "",
                "status":     f"DEMO (latest: {latest.date()})",
                "source":     "demo",
            })
    logger.info(f"Fallback: {len(fixtures)} demo fixtures from most recent matchday")
    return fixtures


def _fuzzy_match(name, pool):
    if name in pool: return name
    lm = {t.lower(): t for t in pool}
    if name.lower() in lm: return lm[name.lower()]
    for t in pool:
        if name.lower() in t.lower() or t.lower() in name.lower(): return t
    words = set(name.lower().split())
    best, bs = None, 0
    for t in pool:
        overlap = len(words & set(t.lower().split()))
        if overlap > bs: bs, best = overlap, t
    return best if bs >= 1 else None


def _sanitise_cols(df):
    seen, nc = {}, {}
    for col in df.columns:
        c = re.sub(r'[^a-zA-Z0-9_]', '_', str(col))
        if c in seen: seen[c] += 1; c = f"{c}_{seen[c]}"
        else: seen[c] = 0
        nc[col] = c
    return df.rename(columns=nc)


def predict_fixture(features_df, model, feature_names,
                    home_team, away_team, league_key, match_date=None):
    """Predict outcome probabilities for a single fixture."""
    sub = features_df[features_df["league_key"] == league_key].copy()
    if match_date:
        sub = sub[sub["date"] < pd.to_datetime(match_date)]

    all_teams = list(
        set(sub["home_team"].unique()) | set(sub["away_team"].unique())
    )
    ht = _fuzzy_match(home_team, all_teams)
    at = _fuzzy_match(away_team, all_teams)
    if not ht or not at:
        return None

    h2h = sub[
        ((sub["home_team"] == ht) & (sub["away_team"] == at)) |
        ((sub["home_team"] == at) & (sub["away_team"] == ht))
    ]
    if not h2h.empty:
        row = h2h.sort_values("date").iloc[[-1]].copy()
        if row.iloc[0]["home_team"] != ht:
            for col in [c for c in row.columns if c.startswith("home_")]:
                ac = col.replace("home_", "away_", 1)
                if ac in row.columns:
                    row[[col, ac]] = row[[ac, col]].values
    else:
        hr = sub[sub["home_team"] == ht]
        ar = sub[sub["away_team"] == at]
        if hr.empty or ar.empty:
            return None
        row = hr.sort_values("date").iloc[[-1]].copy()
        al  = ar.sort_values("date").iloc[-1]
        for col in row.columns:
            if col.startswith("away_"):
                row[col] = al.get(col, row[col].values[0])

    row = _sanitise_cols(row)
    for col in feature_names:
        if col not in row.columns:
            row[col] = -999.0

    X     = row[feature_names].astype(np.float32).fillna(-999)
    proba = model.predict_proba(X)[0]
    top_i = int(np.argmax(proba))
    top_p = float(proba[top_i])
    labels = ["Home Win", "Draw", "Away Win"]
    conf   = "HIGH" if top_p >= 0.65 else "MEDIUM" if top_p >= 0.50 else "LOW"

    return {
        "home_team":       ht,
        "away_team":       at,
        "league_key":      league_key,
        "p_home":          float(proba[0]),
        "p_draw":          float(proba[1]),
        "p_away":          float(proba[2]),
        "top_outcome":     labels[top_i],
        "top_probability": round(top_p, 4),
        "confidence":      conf,
    }


def kelly_stake(p, odds, bankroll, fraction=0.25, max_pct=0.05):
    b = odds - 1.0
    if b <= 0: return 0.0
    fk = (b * p - (1 - p)) / b
    if fk <= 0: return 0.0
    return round(min(fraction * fk, max_pct) * bankroll, 2)


# ════════════════════════════════════════════════════════════════════
# PIPELINE ORCHESTRATOR
# ════════════════════════════════════════════════════════════════════

def run(skip_download=False, skip_train=False,
        league_key=None, api_key=None, bankroll=1000.0):

    PATHS.create_all()
    api_key = load_api_key(api_key)

    # ── 1 & 2: Fetch + build ──────────────────────────────────────────
    if not skip_download:
        print(f"\n{'═'*60}")
        print("  STEP 1/4  Fetching match data from football-data.org")
        print(f"{'═'*60}")
        api_df = fetch_all_results(api_key, league_key=league_key)

        print(f"\n{'═'*60}")
        print("  STEP 2/4  Building master + engineering features")
        print(f"{'═'*60}")
        master      = build_master(api_df)
        features_df = engineer_features(master)
    else:
        fp = PATHS.PROCESSED / "features_master.parquet"
        mp = PATHS.PROCESSED / "master.parquet"
        p  = fp if fp.exists() else mp
        if not p.exists():
            logger.error(
                "No processed data found. Run without --predict-only first."
            )
            return
        features_df = pd.read_parquet(p, engine="pyarrow")
        features_df["date"] = pd.to_datetime(features_df["date"])
        logger.info(f"Loaded existing features: {len(features_df):,} rows")

    # ── 3: Train ──────────────────────────────────────────────────────
    if not skip_train:
        print(f"\n{'═'*60}")
        print("  STEP 3/4  Training Oracle model")
        print(f"{'═'*60}")
        model, feature_names = train_oracle(features_df)
    else:
        mp = PATHS.MODEL_MATCH_RESULT
        fp = PATHS.MODELS / "match_result_features.pkl"
        if not mp.exists():
            logger.error("No model found. Run without --predict-only first.")
            return
        model         = joblib.load(mp)
        feature_names = joblib.load(fp)
        logger.info(f"Loaded existing model ({len(feature_names)} features)")

    # ── 4: Predict ────────────────────────────────────────────────────
    print(f"\n{'═'*60}")
    print("  STEP 4/4  Fetching fixtures & generating predictions")
    print(f"{'═'*60}\n")

    fixtures = fetch_todays_fixtures(api_key, league_key)
    if not fixtures:
        logger.warning("No fixtures to predict.")
        return

    is_live = any(f.get("source") == "live" for f in fixtures)
    cutoff  = TODAY if is_live else None

    results = []
    for fix in fixtures:
        try:
            pred = predict_fixture(
                features_df, model, feature_names,
                fix["home_team"], fix["away_team"], fix["league_key"],
                match_date=cutoff,
            )
            if pred:
                pred["fixture"] = fix
                pred["stake"]   = kelly_stake(
                    pred["top_probability"], 2.10, bankroll
                )
                results.append(pred)
        except Exception as e:
            logger.warning(
                f"Prediction failed {fix['home_team']} vs "
                f"{fix['away_team']}: {e}"
            )

    if not results:
        logger.warning("No predictions generated.")
        return

    results.sort(key=lambda x: (
        {"HIGH": 0, "MEDIUM": 1, "LOW": 2}.get(x["confidence"], 2),
        -x["top_probability"],
    ))

    date_lbl = TODAY if is_live else "DEMO — most recent matchday"
    print(f"{'═'*80}")
    print(f"  🔮 THE ORACLE OF THE PITCH — {date_lbl}")
    if not is_live:
        print(
            "  ⚠  Add API key for live fixtures: "
            "https://www.football-data.org/client/register"
        )
    print(f"{'═'*80}")
    print(
        f"  {'':5}  {'HOME':<22} {'AWAY':<22} {'LGE':<10} "
        f"{'PICK':<12} {'CONF':>4}  {'PROB':>5}  {'STAKE':>6}"
    )
    print(f"  {'─'*78}")

    for pred in results:
        icon = {"HIGH": "🟢", "MEDIUM": "🟡", "LOW": "🔴"}.get(
            pred["confidence"], "⚪"
        )
        print(
            f"  {icon}    {pred['home_team']:<22} {pred['away_team']:<22} "
            f"{pred['league_key']:<10} {pred['top_outcome']:<12} "
            f"{pred['confidence']:>4}  {pred['top_probability']:>4.0%}  "
            f"{pred['stake']:>5.0f}u"
        )

    highs   = [r for r in results if r["confidence"] == "HIGH"]
    mediums = [r for r in results if r["confidence"] == "MEDIUM"]
    lows    = [r for r in results if r["confidence"] == "LOW"]

    print(f"{'═'*80}")
    print(
        f"\n  Predictions: {len(results)}  |  "
        f"🟢 HIGH: {len(highs)}  🟡 MEDIUM: {len(mediums)}  🔴 LOW: {len(lows)}"
    )

    if highs:
        best = highs[0]
        fix  = best["fixture"]
        print(f"\n{'─'*55}")
        print("  🏆 BEST BET OF THE DAY")
        print(f"{'─'*55}")
        print(f"  {best['home_team']}  vs  {best['away_team']}  ({best['league_key']})")
        if fix.get("kickoff"):
            print(f"  Kickoff : {fix['kickoff']} UTC")
        print(f"  Pick    : {best['top_outcome']}")
        print(f"  Confidence: {best['confidence']}  ({best['top_probability']:.1%})")
        print(f"  Kelly stake (25%): {best['stake']:.0f} units of {bankroll:.0f}")
        print("\n  Probabilities:")
        for lbl, p in [
            ("Home Win", best["p_home"]),
            ("Draw",     best["p_draw"]),
            ("Away Win", best["p_away"]),
        ]:
            bar = "█" * int(p * 30) + "░" * (30 - int(p * 30))
            print(f"    {lbl:<12} {p:>5.1%}  {bar}")
        print(f"{'─'*55}\n")

    return results


# ════════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="The Oracle of the Pitch — API-only pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/update_data.py                        # full run
  python src/update_data.py --no-train             # refresh data, keep model
  python src/update_data.py --predict-only         # skip data/train, just predict
  python src/update_data.py --league EPL           # EPL only
  python src/update_data.py --api-key YOUR_KEY     # provide key inline
        """
    )
    parser.add_argument("--no-download",  action="store_true",
                        help="Skip API download")
    parser.add_argument("--no-train",     action="store_true",
                        help="Skip model retrain")
    parser.add_argument("--predict-only", action="store_true",
                        help="Only predict (no download/train)")
    parser.add_argument("--league", default=None,
                        choices=list(LEAGUES.keys()),
                        help="Filter to one league")
    parser.add_argument("--api-key", default=None,
                        help="football-data.org API key")
    parser.add_argument("--bankroll", type=float, default=1000.0,
                        help="Bankroll in units for Kelly sizing (default: 1000)")
    args = parser.parse_args()

    run(
        skip_download = args.predict_only or args.no_download,
        skip_train    = args.predict_only or args.no_train,
        league_key    = args.league,
        api_key       = args.api_key,
        bankroll      = args.bankroll,
    )