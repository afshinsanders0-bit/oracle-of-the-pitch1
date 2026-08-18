"""
app.py — FootballIQ  |  The Oracle of the Pitch
=================================================
ARCHITECTURE CHANGE (v2):
  • All predictions driven by live football-data.org API fixtures — no CSV/parquet needed
    for the "Today's Predictions" and "Analyst Preview" pages.
  • Feature parquet (from update_data.py) is ONLY used by Match Deep Dive for
    rolling-form features when the user explicitly wants the full bet builder.
  • Analyst module upgraded: new system prompt covering Team Over 0.5, Over 1.5,
    BTTS, Over 2.5+BTTS, Corners, Cards, Handicaps, Player props.
  • New "Analyst Preview" page: pick any today fixture → get full LLM preview.
  • Bankroll Simulator and League Analytics require the parquet but degrade gracefully.

Run: streamlit run app.py
Requires: FOOTBALL_DATA_KEY in .env or environment
          Optional: GROQ_API_KEY for analyst previews
"""

import sys, re, os, datetime
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st  # type: ignore[import-untyped]
import pandas as pd
import numpy as np
import joblib
import requests
import time
import plotly.graph_objects as go  # type: ignore[import-untyped]

from config import PATHS, LEAGUES, VALUE_BET, KELLY, SEASONS

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FootballIQ  ⚽",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500;600&display=swap');

html,body,[class*="css"]          { font-family:'DM Sans',sans-serif; }
.main,[data-testid="stAppViewContainer"]{ background:#07090f; }
[data-testid="stSidebar"]         { background:#0a0d17; border-right:1px solid #1a2235; }

/* typography */
h1,h2,h3                          { font-family:'DM Serif Display',serif; color:#edf1f7!important; letter-spacing:-.3px; }
p,li,label,[data-testid="stMarkdownContainer"] { color:#8a9bb5!important; }

/* metric card */
.metric-card{
  background:linear-gradient(145deg,#0d1525,#080c18);
  border:1px solid #1e2d45; border-radius:14px;
  padding:22px 26px; margin:6px 0;
}
.metric-value{ font-size:2rem; font-weight:700; color:#5bc8f5;
               font-family:'DM Mono',monospace; }
.metric-label{ font-size:.7rem; color:#4a6280; text-transform:uppercase;
               letter-spacing:2px; margin-top:6px; }

/* match card */
.match-card{
  background:#0a0f1c; border:1px solid #1a2a40;
  border-radius:12px; padding:16px 20px; margin:8px 0;
  transition:border-color .2s;
}
.match-card:hover{ border-color:#2a4a70; }
.match-title{ font-size:1.05rem; font-weight:600; color:#dde4f0; }
.match-meta { font-size:.78rem; color:#4a6280; margin-top:2px; }

/* badges */
.badge-high  { background:#0a2e1a; color:#3de880; border:1px solid #145c28;
               padding:3px 10px; border-radius:20px; font-size:.72rem; font-weight:600; }
.badge-medium{ background:#2a2000; color:#f5c542; border:1px solid #7a5000;
               padding:3px 10px; border-radius:20px; font-size:.72rem; font-weight:600; }
.badge-low   { background:#2a0e0e; color:#f07070; border:1px solid #7a1a1a;
               padding:3px 10px; border-radius:20px; font-size:.72rem; font-weight:600; }

/* prob bars */
.prob-bar-wrap  { display:flex; align-items:center; gap:10px; margin:4px 0; }
.prob-label     { width:80px; font-size:.78rem; color:#7a90a8; }
.prob-bar-bg    { flex:1; background:#111c30; border-radius:4px; height:7px; }
.prob-bar-fill  { height:7px; border-radius:4px; }
.prob-pct       { width:40px; text-align:right; font-size:.78rem;
                  font-family:'DM Mono',monospace; color:#dde4f0; }

/* value alerts */
.value-alert{
  background:linear-gradient(135deg,#081e10,#050f08);
  border:1px solid #0f6b25; border-left:4px solid #1de860;
  border-radius:8px; padding:14px 18px; margin:8px 0;
}
.value-alert-title{ color:#3de880; font-weight:700; font-size:.9rem; }
.value-alert-body { color:#7ae8a0; font-size:.82rem; margin-top:5px; }

/* analyst card */
.analyst-card{
  background:#08111e; border:1px solid #1a3050;
  border-left:4px solid #5bc8f5;
  border-radius:10px; padding:22px 26px; margin:14px 0;
  font-size:.9rem; line-height:1.75; color:#c0cfe0!important;
  white-space:pre-wrap;
}
.analyst-card strong{ color:#5bc8f5!important; }

/* bet builder card */
.bet-builder-card{
  background:#060d1b; border:1px solid #1a3558;
  border-radius:14px; padding:20px 24px; margin:12px 0;
}
.bet-builder-title{
  font-size:1rem; font-weight:700; color:#5bc8f5;
  margin-bottom:14px; border-bottom:1px solid #1a2e48;
  padding-bottom:10px;
}
.market-row{
  display:flex; justify-content:space-between; align-items:center;
  padding:9px 0; border-bottom:1px solid #0e1b2e;
}
.market-name { font-size:.82rem; color:#4a6280; width:160px; }
.market-pick { font-size:.88rem; font-weight:600; color:#dde4f0; }
.acca-card{
  background:#080f0a; border:1px solid #153822;
  border-radius:10px; padding:16px 20px; margin:8px 0;
}
.acca-title   { color:#3de880; font-weight:700; font-size:.86rem; margin-bottom:8px; }
.acca-odds    { font-family:'DM Mono',monospace; font-size:1.35rem;
                color:#5bc8f5; font-weight:700; }

/* section header */
.section-header{
  border-bottom:1px solid #1a2a40; padding-bottom:12px;
  margin-bottom:20px; color:#dde4f0;
  font-family:'DM Serif Display',serif; font-size:1.4rem;
}

/* api setup */
.api-setup-box{
  background:#0a0f1c; border:1px solid #1a3050;
  border-left:4px solid #5bc8f5; border-radius:8px;
  padding:20px 24px; margin:16px 0;
}
.api-setup-title{ color:#5bc8f5; font-weight:700; font-size:.95rem; margin-bottom:8px; }
.api-setup-step { color:#7a90a8; font-size:.85rem; padding:3px 0; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# SHARED HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def get_api_key() -> str | None:
    key = os.environ.get("FOOTBALL_DATA_KEY", "").strip()
    if key:
        return key
    env = Path(__file__).parent / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if line.startswith("FOOTBALL_DATA_KEY="):
                k = line.split("=", 1)[1].strip().strip('"\'')
                if k:
                    return k
    return None


def get_groq_key() -> str | None:
    key = os.environ.get("GROQ_API_KEY", "").strip()
    if key:
        return key
    env = Path(__file__).parent / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if line.startswith("GROQ_API_KEY="):
                k = line.split("=", 1)[1].strip().strip('"\'')
                if k:
                    return k
    return None


def show_api_setup(key_name: str = "FOOTBALL_DATA_KEY"):
    """Show API setup instructions when key is missing."""
    if key_name == "FOOTBALL_DATA_KEY":
        st.markdown("""
        <div class="api-setup-box">
          <div class="api-setup-title">🔑 football-data.org API Key Required</div>
          <div class="api-setup-step">Free, no credit card needed.</div><br>
          <div class="api-setup-step"><strong style="color:#dde4f0;">Step 1</strong> →
            Register at <a href="https://www.football-data.org/client/register" target="_blank"
            style="color:#5bc8f5;">football-data.org/client/register</a></div>
          <div class="api-setup-step"><strong style="color:#dde4f0;">Step 2</strong> →
            Add to your .env file:</div>
          <pre style="background:#040810;color:#3de880;padding:10px 14px;
                      border-radius:6px;margin-top:8px;font-size:.82rem;">
FOOTBALL_DATA_KEY=your_key_here</pre>
        </div>
        """, unsafe_allow_html=True)
        temp = st.text_input("Or paste key here (session only)", type="password",
                             placeholder="abc123…")
        if temp:
            os.environ["FOOTBALL_DATA_KEY"] = temp.strip()
            st.success("✓ Key set — refreshing…")
            st.rerun()
    else:
        st.markdown("""
        <div class="api-setup-box">
          <div class="api-setup-title">🔑 GROQ API Key Required for Analyst</div>
          <div class="api-setup-step">Free key at
            <a href="https://console.groq.com" target="_blank" style="color:#5bc8f5;">
            console.groq.com</a></div>
          <pre style="background:#040810;color:#3de880;padding:10px 14px;
                      border-radius:6px;margin-top:8px;font-size:.82rem;">
GROQ_API_KEY=your_key_here</pre>
        </div>
        """, unsafe_allow_html=True)
        temp = st.text_input("Or paste Groq key (session only)", type="password",
                             placeholder="gsk_…")
        if temp:
            os.environ["GROQ_API_KEY"] = temp.strip()
            st.success("✓ Groq key set — refreshing…")
            st.rerun()


# ── Fetch live fixtures (cached 5 min) ────────────────────────────────────────

# Maps our league keys → football-data.org v4 competition codes
# Keys MUST match config.LEAGUES exactly
FD_ORG_COMP = {
    # Domestic — available on current plan
    "EPL":            "PL",
    "ENG_CHAMP":      "ELC",
    "LA_LIGA":        "PD",
    "SERIE_A":        "SA",
    "LIGUE_1":        "FL1",
    "BUNDESLIGA":     "BL1",
}

TEAM_NORM = {
    # EPL
    "Arsenal FC":"Arsenal","Chelsea FC":"Chelsea","Liverpool FC":"Liverpool",
    "Manchester City FC":"Man City","Manchester United FC":"Man United",
    "Tottenham Hotspur FC":"Tottenham","Newcastle United FC":"Newcastle",
    "Aston Villa FC":"Aston Villa","West Ham United FC":"West Ham",
    "Brighton & Hove Albion FC":"Brighton","Brentford FC":"Brentford",
    "Fulham FC":"Fulham","Crystal Palace FC":"Crystal Palace",
    "Wolverhampton Wanderers FC":"Wolves","Everton FC":"Everton",
    "Nottingham Forest FC":"Nott'm Forest","AFC Bournemouth":"Bournemouth",
    "Leicester City FC":"Leicester","Ipswich Town FC":"Ipswich",
    "Southampton FC":"Southampton","Leeds United FC":"Leeds",
    "Burnley FC":"Burnley","Sheffield United FC":"Sheffield United","Luton Town FC":"Luton",
    # La Liga
    "FC Barcelona":"Barcelona","Real Madrid CF":"Real Madrid",
    "Club Atletico de Madrid":"Ath Madrid","Club Atlético de Madrid":"Ath Madrid",
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
    # Serie A
    "Inter Milan":"Inter","AC Milan":"Milan","Juventus FC":"Juventus",
    "SSC Napoli":"Napoli","AS Roma":"Roma","SS Lazio":"Lazio",
    "ACF Fiorentina":"Fiorentina","Atalanta BC":"Atalanta",
    "Bologna FC 1909":"Bologna","Torino FC":"Torino","Udinese Calcio":"Udinese",
    "Cagliari Calcio":"Cagliari","Genoa CFC":"Genoa","US Lecce":"Lecce",
    "Hellas Verona FC":"Verona","Empoli FC":"Empoli","Venezia FC":"Venezia",
    "AC Monza":"Monza","Parma Calcio 1913":"Parma","Como 1907":"Como",
    # Bundesliga
    "FC Bayern Munchen":"Bayern Munich","FC Bayern München":"Bayern Munich",
    "Borussia Dortmund":"Dortmund","Bayer 04 Leverkusen":"Leverkusen",
    "Eintracht Frankfurt":"Ein Frankfurt",
    "SC Freiburg":"Freiburg","VfB Stuttgart":"Stuttgart",
    "1. FC Union Berlin":"Union Berlin","1. FSV Mainz 05":"Mainz",
    "Borussia Monchengladbach":"M'gladbach","Borussia Mönchengladbach":"M'gladbach",
    "TSG 1899 Hoffenheim":"Hoffenheim","FC Augsburg":"Augsburg",
    "VfL Wolfsburg":"Wolfsburg",
    "VfL Bochum 1848":"Bochum","1. FC Heidenheim 1846":"Heidenheim",
    "FC St. Pauli 1910":"St Pauli",
    # Ligue 1
    "Paris Saint-Germain FC":"Paris SG","Olympique de Marseille":"Marseille",
    "Olympique Lyonnais":"Lyon","AS Monaco FC":"Monaco","LOSC Lille":"Lille",
    "RC Strasbourg Alsace":"Strasbourg","OGC Nice":"Nice",
    "Stade Rennais FC 1901":"Rennes","RC Lens":"Lens",
    "Stade Brestois 29":"Brest","Montpellier HSC":"Montpellier",
    "FC Nantes":"Nantes","Toulouse FC":"Toulouse","Le Havre AC":"Le Havre",
    "AJ Auxerre":"Auxerre","Angers SCO":"Angers",
    "AS Saint-Etienne":"St Etienne","AS Saint-Étienne":"St Etienne",
    "Stade de Reims":"Reims",
    # Championship
    "Middlesbrough FC":"Middlesbrough","Coventry City FC":"Coventry","Watford FC":"Watford",
    "Preston North End FC":"Preston","Millwall FC":"Millwall",
    "Swansea City AFC":"Swansea","Cardiff City FC":"Cardiff","Wrexham AFC":"Wrexham",
    "Bristol City FC":"Bristol City","Norwich City FC":"Norwich",
    "Stoke City FC":"Stoke","Hull City AFC":"Hull",
    "Queens Park Rangers FC":"QPR","West Bromwich Albion FC":"West Brom",
    "Blackburn Rovers FC":"Blackburn","Plymouth Argyle FC":"Plymouth",
    "Sunderland AFC":"Sunderland","Derby County FC":"Derby",
    "Portsmouth FC":"Portsmouth","Oxford United FC":"Oxford",
}


@st.cache_data(ttl=300)
def fetch_fixtures_today(
    api_key: str | None,                # FIX: was `str`, now accepts None
    league_key: str | None = None,
) -> tuple[list, list]:
    today   = datetime.datetime.now(datetime.timezone.utc).date().isoformat()
    if league_key and league_key in FD_ORG_COMP:
        leagues_to_fetch = {league_key: FD_ORG_COMP[league_key]}
    else:
        leagues_to_fetch = FD_ORG_COMP
    headers  = {"X-Auth-Token": api_key} if api_key else {}
    fixtures, errors = [], []

    for lk, code in leagues_to_fetch.items():
        if not code:
            continue
        try:
            time.sleep(1.2)
            url  = (f"https://api.football-data.org/v4/competitions/{code}/matches"
                    f"?dateFrom={today}&dateTo={today}&status=SCHEDULED,TIMED,IN_PLAY,LIVE")
            resp = requests.get(url, headers=headers, timeout=12)

            if resp.status_code == 200:
                for m in resp.json().get("matches", []):
                    ht_raw = m["homeTeam"]["name"]
                    at_raw = m["awayTeam"]["name"]
                    score = m.get("score", {})
                    fixtures.append({
                        "home_team":       TEAM_NORM.get(ht_raw, ht_raw),
                        "away_team":       TEAM_NORM.get(at_raw, at_raw),
                        "home_raw":        ht_raw,
                        "away_raw":        at_raw,
                        "league_key":      lk,
                        "kickoff":         m.get("utcDate", "")[:16],
                        "status":          m.get("status", "SCHEDULED"),
                        "matchday":        m.get("matchday", ""),
                        "home_score":      score.get("home"),
                        "away_score":      score.get("away"),
                        "score_duration":  score.get("duration", ""),
                        "minute":          m.get("minute"),
                    })
            elif resp.status_code == 403:
                errors.append(f"{lk}: Not available on your plan")
            elif resp.status_code == 404:
                errors.append(f"{lk}: Competition code not found")
            elif resp.status_code == 429:
                errors.append(f"{lk}: Rate limited — refresh in 60s")
            else:
                errors.append(f"{lk}: HTTP {resp.status_code}")
        except Exception as e:
            errors.append(f"{lk}: {e}")

    return fixtures, errors


# ── Load model (lazy) ─────────────────────────────────────────────────────────

@st.cache_resource
def load_model_cached(market: str = "match_result"):
    # Feature file candidates — pipeline may save under different names
    feat_candidates = {
        "match_result": [
            "match_result_features.pkl",
            "match_result_current_features.pkl",
        ],
        "btts":       ["btts_features.pkl"],
        "over_under": ["over_under_features.pkl"],
        "corners":    ["corners_features.pkl"],
    }
    model_files = {
        "match_result": PATHS.MODEL_MATCH_RESULT,
        "btts":         PATHS.MODEL_BTTS,
        "over_under":   PATHS.MODEL_OVER_UNDER,
        "corners":      PATHS.MODEL_CORNERS,
    }
    mp = model_files.get(market)
    if not mp or not mp.exists():
        return None, None
    try:
        model = joblib.load(mp)
    except Exception:
        return None, None

    # Try each candidate feature file
    feats = None
    for fname in feat_candidates.get(market, []):
        fp = PATHS.MODELS / fname
        if fp.exists():
            try:
                feats = joblib.load(fp)
                break
            except Exception:
                continue

    # Last resort: get feature names from model itself
    if feats is None:
        try:
            feats = list(model.get_booster().feature_names)
        except Exception:
            pass

    return model, feats


# ── Load features parquet (optional — only for Deep Dive) ─────────────────────

@st.cache_data(ttl=3600)
def load_features_cached() -> pd.DataFrame | None:
    candidates = [
        "features_master.parquet",
        "master.parquet",
        "master_with_elo.parquet",
        "features.parquet",
    ]
    for fname in candidates:
        p = PATHS.PROCESSED / fname
        if p.exists():
            try:
                df = pd.read_parquet(p, engine="auto")
                df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None) \
                             if pd.to_datetime(df["date"]).dt.tz is None \
                             else pd.to_datetime(df["date"]).dt.tz_convert("UTC").dt.tz_localize(None)
                df.columns = [re.sub(r'[^a-zA-Z0-9_]', '_', str(c)) for c in df.columns]
                return df
            except Exception:
                continue

    league_files = sorted(PATHS.PROCESSED.glob("*_master.parquet"))
    if league_files:
        try:
            dfs = []
            for p in league_files:
                df = pd.read_parquet(p, engine="auto")
                df.columns = [re.sub(r'[^a-zA-Z0-9_]', '_', str(c)) for c in df.columns]
                dfs.append(df)
            combined = pd.concat(dfs, ignore_index=True)
            combined["date"] = pd.to_datetime(combined["date"]).dt.tz_localize(None) \
                               if pd.to_datetime(combined["date"]).dt.tz is None \
                               else pd.to_datetime(combined["date"]).dt.tz_convert("UTC").dt.tz_localize(None)
            combined = combined.sort_values("date").reset_index(drop=True)
            return combined
        except Exception:
            pass

    return None


# ── Mini helpers ──────────────────────────────────────────────────────────────

def sanitise_cols(df: pd.DataFrame) -> pd.DataFrame:
    seen, nc = {}, {}
    for col in df.columns:
        c = re.sub(r'[^a-zA-Z0-9_]', '_', str(col))
        if c in seen:
            seen[c] += 1; c = f"{c}_{seen[c]}"
        else:
            seen[c] = 0
        nc[col] = c
    return df.rename(columns=nc)


def fuzzy_team(name: str, pool: list) -> str | None:
    if name in pool: return name
    lm = {t.lower(): t for t in pool}
    if name.lower() in lm: return lm[name.lower()]
    for t in pool:
        if name.lower() in t.lower(): return t
    words = set(name.lower().split())
    best, bs, best_t = None, 0, None
    for t in pool:
        tw = set(t.lower().split())
        ov = len(words & tw)
        if ov > bs: bs, best_t = ov, t
    if best_t is not None and bs >= 2:
        return best_t
    return None


def badge(conf: str) -> str:
    cls  = {"HIGH":"badge-high","MEDIUM":"badge-medium","LOW":"badge-low"}.get(conf,"badge-low")
    icon = {"HIGH":"🟢","MEDIUM":"🟡","LOW":"🔴"}.get(conf,"⚪")
    return f'<span class="{cls}">{icon} {conf}</span>'


def prob_bars(p_home: float, p_draw: float, p_away: float) -> str:
    html = ""
    for label, prob, color in [
        ("Home Win", p_home, "#5bc8f5"),
        ("Draw",     p_draw, "#f5c542"),
        ("Away Win", p_away, "#f07070"),
    ]:
        w = int(prob * 100)
        html += f"""
        <div class="prob-bar-wrap">
          <div class="prob-label">{label}</div>
          <div class="prob-bar-bg"><div class="prob-bar-fill"
               style="width:{w}%;background:{color};"></div></div>
          <div class="prob-pct">{prob:.0%}</div>
        </div>"""
    return html


def prob_bars_generic(probs: dict) -> str:
    colors = ["#5bc8f5","#f5c542","#f07070","#b07efa"]
    html   = ""
    for i, (label, prob) in enumerate(probs.items()):
        c = colors[i % len(colors)]
        w = int(prob * 100)
        html += f"""
        <div class="prob-bar-wrap">
          <div class="prob-label" style="width:110px;">{label}</div>
          <div class="prob-bar-bg"><div class="prob-bar-fill"
               style="width:{w}%;background:{c};"></div></div>
          <div class="prob-pct">{prob:.0%}</div>
        </div>"""
    return html


def quick_kelly_ui(model_prob: float, odds: float, bankroll: float) -> dict:
    b      = max(odds - 1.0, 0.0)
    p      = model_prob
    q      = 1.0 - p
    full_k = max(0.0, (b * p - q) / b) if b > 0 else 0.0
    frac_k = min(full_k * KELLY.FRACTION, KELLY.MAX_STAKE_PCT)
    return {
        "stake_units": round(frac_k * bankroll, 2),
        "stake_pct":   round(frac_k * 100, 2),
        "ev":          round((p * odds) - 1.0, 4),
    }


def predict_one_match_live(df, home, away, league_key, today):
    """
    Predict one match. Bulletproof version:
    - Falls back through multiple team/league lookup strategies
    - Cross-league fallback: if the fixture league has no data, use any team
      found globally (the model only cares about rolling features, not the league)
    - Timezone-safe date comparison: strips tz from the date column if present
    - Fills all missing features with -999 (model handles gracefully)
    - Never raises — always returns dict or None
    """
    model, feature_names = load_model_cached("match_result")
    if model is None or feature_names is None:
        return None
    if df is None or df.empty:
        return None

    # ── FIX 1: timezone-safe today_dt ─────────────────────────────────────
    # The parquet may store dates as datetime64[ns, UTC]. pd.to_datetime(today)
    # produces a tz-naive Timestamp, which causes "Invalid comparison" errors.
    # Solution: make today_dt match the tz of the date column.
    raw_today = pd.to_datetime(today)
    if df["date"].dt.tz is not None:
        today_dt = raw_today.tz_localize("UTC")
    else:
        today_dt = raw_today

    all_teams_global = list(set(df["home_team"].tolist()) | set(df["away_team"].tolist()))

    # ── FIX 2: cross-league team lookup ───────────────────────────────────
    # If the fixture's league (e.g. EPL) has no rows in the parquet, the
    # league_teams list is empty and fuzzy_team always returns None.
    # Instead: try league-first, then fall back immediately to the full pool.
    sub_league   = df[df["league_key"] == league_key]
    league_teams = list(set(sub_league["home_team"].tolist()) | set(sub_league["away_team"].tolist()))

    ht = fuzzy_team(home, league_teams) or fuzzy_team(home, all_teams_global)
    at = fuzzy_team(away, league_teams) or fuzzy_team(away, all_teams_global)

    if not ht or not at:
        return None

    # ── Step 2: Get the data subset ───────────────────────────────────────
    # When teams were matched globally (cross-league), search across ALL data.
    def get_sub(team_a, team_b):
        mask = (
            (df["home_team"].isin([team_a, team_b])) |
            (df["away_team"].isin([team_a, team_b]))
        )
        sub_all  = df[mask].copy()
        sub_past = sub_all[sub_all["date"] < today_dt]
        return sub_past if not sub_past.empty else sub_all

    sub = get_sub(ht, at)
    if sub.empty:
        return None

    # ── Step 3: Build the feature row ─────────────────────────────────────
    h2h = sub[
        ((sub["home_team"] == ht) & (sub["away_team"] == at)) |
        ((sub["home_team"] == at) & (sub["away_team"] == ht))
    ].sort_values("date")

    if not h2h.empty:
        row = h2h.iloc[[-1]].copy()
        if row.iloc[0]["home_team"] != ht:
            swap_cols = [c for c in row.columns if c.startswith("home_")]
            for c in swap_cols:
                ac = c.replace("home_", "away_", 1)
                if ac in row.columns:
                    row[c], row[ac] = row[ac].values.copy(), row[c].values.copy()
            row["home_team"] = ht
            row["away_team"] = at
    else:
        hr = sub[sub["home_team"] == ht]
        if hr.empty:
            hr = sub[(sub["home_team"] == ht) | (sub["away_team"] == ht)]
        ar = sub[sub["away_team"] == at]
        if ar.empty:
            ar = sub[(sub["home_team"] == at) | (sub["away_team"] == at)]

        if hr.empty and ar.empty:
            row = sub.sort_values("date").iloc[[-1]].copy()
        elif hr.empty:
            row = ar.sort_values("date").iloc[[-1]].copy()
        elif ar.empty:
            row = hr.sort_values("date").iloc[[-1]].copy()
        else:
            row = hr.sort_values("date").iloc[[-1]].copy()
            al  = ar.sort_values("date").iloc[-1]
            for col in row.columns:
                if col.startswith("away_"):
                    try:
                        row[col] = al[col]
                    except (KeyError, TypeError):
                        pass
        row = row.copy()
        row["home_team"] = ht
        row["away_team"] = at
        row["league_key"] = league_key

    # ── Step 4: Sanitise + fill + predict ────────────────────────────────
    row = sanitise_cols(row)
    for col in feature_names:
        if col not in row.columns:
            row[col] = -999.0
    X = row[feature_names].astype(np.float32).fillna(-999)

    proba   = model.predict_proba(X)[0]
    labels  = {0: "Home Win", 1: "Draw", 2: "Away Win"}
    top_idx = int(np.argmax(proba))
    top_p   = float(proba[top_idx])

    btts_prob     = None
    over_25_prob  = None
    exp_hg = exp_ag = None

    btts_model, btts_feats = load_model_cached("btts")
    if btts_model and btts_feats:
        try:
            Xb = row[[f for f in btts_feats if f in row.columns]].astype(np.float32).fillna(-999)
            for f in btts_feats:
                if f not in Xb.columns: Xb[f] = -999.0
            Xb = Xb[btts_feats]
            btts_prob = float(btts_model.predict_proba(Xb)[0][1])
        except Exception:
            pass

    ou_model, ou_feats = load_model_cached("over_under")
    if ou_model and ou_feats:
        try:
            Xu = row[[f for f in ou_feats if f in row.columns]].astype(np.float32).fillna(-999)
            for f in ou_feats:
                if f not in Xu.columns: Xu[f] = -999.0
            Xu = Xu[ou_feats]
            over_25_prob = float(ou_model.predict_proba(Xu)[0][1])
        except Exception:
            pass

    for exp_col in ["expected_total_goals", "expected_home_goals"]:
        if exp_col in row.columns:
            v = float(row[exp_col].values[0])
            if v > 0:
                exp_hg = float(row.get("expected_home_goals", pd.Series([1.4])).values[0])
                exp_ag = float(row.get("expected_away_goals", pd.Series([1.1])).values[0])
                break

    return {
        "home_team":       ht,
        "away_team":       at,
        "league_key":      league_key,
        "p_home":          float(proba[0]),
        "p_draw":          float(proba[1]),
        "p_away":          float(proba[2]),
        "top_outcome":     labels[top_idx],
        "top_probability": round(top_p, 4),
        "confidence":      ("HIGH" if top_p >= 0.65
                            else "MEDIUM" if top_p >= 0.50 else "LOW"),
        "btts_prob":       btts_prob,
        "over_25_prob":    over_25_prob,
        "expected_home_goals": exp_hg,
        "expected_away_goals": exp_ag,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# ALERTS & SIGNAL TRACKER — PERSISTENCE HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

import json, uuid
import datetime

ALERTS_TRACKER_PATH = PATHS.LIVE / "alerts_tracker.json"


def _ensure_live_dirs() -> None:
    PATHS.LIVE.mkdir(parents=True, exist_ok=True)


def load_alerts_tracker() -> dict:
    _ensure_live_dirs()
    if ALERTS_TRACKER_PATH.exists():
        try:
            return json.loads(ALERTS_TRACKER_PATH.read_text())
        except Exception:
            pass
    return {"signals": [], "alerts": [], "tracked": []}


def save_alerts_tracker(data: dict) -> None:
    _ensure_live_dirs()
    ALERTS_TRACKER_PATH.write_text(json.dumps(data, indent=2, default=str))


def generate_signals_from_predictions(predictions: list[dict], market_odds_map: dict | None = None) -> list[dict]:
    """
    Convert raw prediction dicts (from predict_one_match_live) into
    structured Signal objects for the tracker.
    """
    signals = []
    now = datetime.datetime.now().isoformat()
    for pred in predictions:
        fixture_key = f"{pred.get('home_team','')} vs {pred.get('away_team','')}"
        league_key  = pred.get("league_key", "")
        top_outcome = pred.get("top_outcome", "")
        top_prob    = pred.get("top_probability", 0)
        conf        = pred.get("confidence", "LOW")
        ev          = None
        odds_val    = None
        kelly       = None

        if market_odds_map:
            mo = market_odds_map.get("match_result", {})
            odds_val = mo.get(top_outcome)
            if odds_val and top_prob > 0:
                kelly = quick_kelly_ui(top_prob, odds_val, bankroll)
                ev    = kelly.get("ev")

        signal = {
            "id":             str(uuid.uuid4())[:8],
            "date":           now,
            "fixture_key":    fixture_key,
            "home_team":      pred.get("home_team", ""),
            "away_team":      pred.get("away_team", ""),
            "league_key":     league_key,
            "market":         "match_result",
            "outcome":        top_outcome,
            "probability":    round(top_prob, 4),
            "confidence":     conf,
            "ev":             round(ev, 4) if ev is not None else None,
            "kelly_stake":    round(kelly["stake_units"], 2) if kelly else None,
            "odds":           round(odds_val, 2) if odds_val else None,
            "is_value":       bool(ev and ev > 0),
            "source":         "model",
            "status":         "active",
        }
        signals.append(signal)

        if pred.get("btts_prob"):
            signals.append({
                "id":          str(uuid.uuid4())[:8],
                "date":        now,
                "fixture_key": fixture_key,
                "home_team":   pred.get("home_team", ""),
                "away_team":   pred.get("away_team", ""),
                "league_key":  league_key,
                "market":      "btts",
                "outcome":     "Yes",
                "probability": round(float(pred["btts_prob"]), 4),
                "confidence":  conf,
                "ev":          None,
                "kelly_stake": None,
                "odds":        None,
                "is_value":    False,
                "source":      "model",
                "status":      "active",
            })
        if pred.get("over_25_prob"):
            signals.append({
                "id":          str(uuid.uuid4())[:8],
                "date":        now,
                "fixture_key": fixture_key,
                "home_team":   pred.get("home_team", ""),
                "away_team":   pred.get("away_team", ""),
                "league_key":  league_key,
                "market":      "over_under",
                "outcome":     "Over 2.5",
                "probability": round(float(pred["over_25_prob"]), 4),
                "confidence":  conf,
                "ev":          None,
                "kelly_stake": None,
                "odds":        None,
                "is_value":    False,
                "source":      "model",
                "status":      "active",
            })
    return signals


def check_alerts(predictions: list[dict], alerts: list[dict]) -> list[dict]:
    """
    Compare current predictions against saved alerts.
    Returns list of alerts whose condition was met, with triggered_prob filled.
    """
    triggered = []
    pred_map = {f"{p.get('home_team','')} vs {p.get('away_team','')}": p for p in predictions}
    for alert in alerts:
        if alert.get("status") != "active":
            continue
        fix = pred_map.get(alert.get("fixture_key", ""))
        if not fix:
            continue
        market = alert.get("market", "match_result")
        outcome = alert.get("outcome", "")
        condition = alert.get("condition", "above")
        threshold = float(alert.get("threshold", 0))
        prob = None

        if market == "match_result":
            probs = {
                "Home Win": fix.get("p_home", 0),
                "Draw":     fix.get("p_draw", 0),
                "Away Win": fix.get("p_away", 0),
            }
            prob = probs.get(outcome)
        elif market == "btts":
            prob = fix.get("btts_prob")
            if outcome not in ("Yes", "No"):
                outcome = "Yes"
        elif market == "over_under":
            prob = fix.get("over_25_prob")
            if outcome not in ("Over 2.5", "Under 2.5"):
                outcome = "Over 2.5"

        if prob is None:
            continue

        hit = False
        if condition == "above" and prob >= threshold:
            hit = True
        elif condition == "below" and prob <= threshold:
            hit = True

        if hit:
            triggered.append({
                **alert,
                "triggered_prob": round(float(prob), 4),
                "triggered_at":   datetime.datetime.now().isoformat(),
                "status":         "triggered",
            })
    return triggered


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("""
    <div style="padding:20px 0 10px 0;">
      <div style="font-family:'DM Serif Display',serif;font-size:1.65rem;
                  color:#5bc8f5;letter-spacing:-.5px;">⚽ FootballIQ</div>
      <div style="font-size:.7rem;color:#3a5070;margin-top:3px;letter-spacing:2px;">
        THE ORACLE OF THE PITCH</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    page = st.radio("Navigation", [
        "🔴 Live Scores",
        "🎯  Today's Predictions",
        "🧠  Analyst Preview",
        "🔍  Match Deep Dive",
        "🔔  Alerts & Signal Tracker",
        "💰  Bankroll Simulator",
        "📊  League Analytics",
        "⚙️  Model Performance",
    ], label_visibility="collapsed")

    st.markdown("---")
    bankroll = st.number_input("💳 Bankroll (units)", value=1000.0,
                               step=50.0, min_value=1.0)

    st.markdown("---")
    api_key  = get_api_key()
    groq_key = get_groq_key()

    for label, key, colour, sub in [
        ("football-data.org", api_key,  "#0a2e1a", "#3de880"),
        ("Groq (Analyst)",    groq_key, "#1a1a00", "#f5c542"),
    ]:
        connected = bool(key)
        bg   = colour if connected else "#2a0e0e"
        col  = sub if connected else "#f07070"
        icon = "🟢" if connected else "🔴"
        stat = "Connected" if connected else "No Key"
        st.markdown(f"""
        <div style="background:{bg};border:1px solid {col}33;
                    border-radius:8px;padding:8px 12px;margin:4px 0;">
          <div style="color:{col};font-size:.78rem;font-weight:600;">{icon} {label}</div>
          <div style="color:{col}99;font-size:.72rem;">{stat}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div style="font-size:.72rem;color:#2a4060;">Data: football-data.org<br>'
                'ML: XGBoost | Staking: 25% Kelly<br>LLM: Groq llama-3.3-70b</div>',
                unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# LOAD CORE ASSETS
# ═══════════════════════════════════════════════════════════════════════════════

api_key  = get_api_key()
groq_key = get_groq_key()
df       = load_features_cached()    # may be None — pages handle gracefully
model_mr, feat_mr = load_model_cached("match_result")
model_ok = model_mr is not None and feat_mr is not None
data_ok  = df is not None


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 0 — LIVE SCORES
# ═══════════════════════════════════════════════════════════════════════════════

if "Live Scores" in page:
    st.markdown('<div class="section-header">🔴 Live Scores</div>',
                unsafe_allow_html=True)

    if not api_key:
        show_api_setup("FOOTBALL_DATA_KEY")
        st.stop()

    col_r1, col_r2 = st.columns([4, 1])
    league_filter = col_r1.selectbox(
        "Filter by league",
        ["All Leagues"] + list(FD_ORG_COMP.keys()),
        format_func=lambda x: x if x == "All Leagues"
                               else LEAGUES.get(x, {}).get("name", x)
    )
    if col_r2.button("🔄 Refresh", width='stretch'):
        st.cache_data.clear()

    lk_filter = None if league_filter == "All Leagues" else league_filter

    with st.spinner("Fetching live scores…"):
        fixtures, api_errors = fetch_fixtures_today(api_key, lk_filter)

    for err in api_errors:
        st.warning(f"⚠️ {err}")

    if not fixtures:
        st.info("📅 No matches today.")
        st.stop()

    live_fix = [f for f in fixtures if f.get("status") in ("IN_PLAY", "LIVE", "PAUSED")]
    finished_fix = [f for f in fixtures if f.get("status") == "FINISHED"]
    scheduled_fix = [f for f in fixtures if f.get("status") not in ("IN_PLAY", "LIVE", "PAUSED", "FINISHED")]

    if live_fix:
        st.markdown("""
        <div style="background:linear-gradient(90deg,#0a2e1a,#0d1f14);
                    border:1px solid #3de88033;border-radius:10px;padding:14px 18px;margin-bottom:16px;">
          <div style="color:#3de880;font-size:.9rem;font-weight:700;">
            🔴 LIVE NOW
          </div>
        </div>""", unsafe_allow_html=True)

        for fix in live_fix:
            hs = fix.get("home_score")
            as_ = fix.get("away_score")
            minute = fix.get("minute", "")
            status_lbl = {
                "IN_PLAY": f"⚽ {minute}'",
                "LIVE": "🔴 LIVE",
                "PAUSED": "⏸️ HT",
            }.get(fix.get("status", ""), "🔴 LIVE")
            score_str = f"{hs or 0} - {as_ or 0}" if hs is not None and as_ is not None else "vs"

            st.markdown(f"""
            <div style="background:#0a0f1c;border:1px solid #1a2a40;border-radius:10px;
                        padding:14px 18px;margin-bottom:8px;display:flex;
                        align-items:center;justify-content:space-between;">
              <div style="flex:1;">
                <div style="color:#7a90a8;font-size:.75rem;margin-bottom:4px;">
                  {LEAGUES.get(fix['league_key'], {}).get('name', fix['league_key'])}
                </div>
                <div style="color:#dde4f0;font-size:1rem;font-weight:600;">
                  {fix['home_team']} <span style="color:#5bc8f5;">vs</span> {fix['away_team']}
                </div>
              </div>
              <div style="text-align:center;margin:0 24px;">
                <div style="color:#3de880;font-size:1.8rem;font-weight:700;">
                  {score_str}
                </div>
              </div>
              <div style="text-align:right;min-width:80px;">
                <div style="color:#f5c542;font-size:.85rem;font-weight:600;">{status_lbl}</div>
                <div style="color:#7a90a8;font-size:.72rem;">{fix.get('matchday', '')}</div>
              </div>
            </div>""", unsafe_allow_html=True)

        import streamlit.components.v1 as components
        components.html("""
        <script>
        setTimeout(function() { window.location.reload(); }, 30000);
        </script>
        """, height=0)
        st.caption("🔄 Auto-refreshing every 30s while matches are live")

    else:
        st.info("📡 No live matches at the moment.")

    if scheduled_fix:
        with st.expander(f"📅 Upcoming Today ({len(scheduled_fix)})", expanded=False):
            for fix in scheduled_fix:
                kick = fix.get("kickoff", "")[11:16] if fix.get("kickoff") else ""
                st.markdown(f"- **{fix['home_team']}** vs **{fix['away_team']}** "
                            f"({LEAGUES.get(fix['league_key'], {}).get('name', fix['league_key'])}) "
                            f"{'• ' + kick + ' UTC' if kick else ''}")

    if finished_fix:
        with st.expander(f"✅ Finished Today ({len(finished_fix)})", expanded=False):
            for fix in finished_fix:
                hs = fix.get("home_score", "?")
                as_ = fix.get("away_score", "?")
                st.markdown(f"- **{fix['home_team']}** {hs} - {as_} **{fix['away_team']}** "
                            f"({LEAGUES.get(fix['league_key'], {}).get('name', fix['league_key'])})")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — TODAY'S PREDICTIONS  (100% API-driven, no parquet needed)
# ═══════════════════════════════════════════════════════════════════════════════

if "Today" in page:
    today = datetime.datetime.now(datetime.timezone.utc).date().isoformat()
    st.markdown(f'<div class="section-header">🎯 Today\'s Predictions — {today}</div>',
                unsafe_allow_html=True)

    if not api_key:
        show_api_setup("FOOTBALL_DATA_KEY")
        st.stop()

    col_f1, col_f2, col_f3 = st.columns([3, 2, 1])
    league_filter = col_f1.selectbox(
        "Filter by league",
        ["All Leagues"] + list(FD_ORG_COMP.keys()),
        format_func=lambda x: x if x == "All Leagues"
                               else LEAGUES.get(x, {}).get("name", x)
    )
    show_no_model = col_f2.checkbox("Show fixtures even without model", value=True)
    if col_f3.button("🔄 Refresh", width='stretch'):
        st.cache_data.clear()

    lk_filter = None if league_filter == "All Leagues" else league_filter

    with st.spinner("Fetching today's fixtures…"):
        fixtures, api_errors = fetch_fixtures_today(api_key, lk_filter)

    for err in api_errors:
        st.warning(f"⚠️ {err}")

    if not fixtures:
        st.info(f"📅 No matches scheduled today ({today}).")
        st.stop()

    # ── Run predictions if model is available ─────────────────────────────
    predictions  = []
    no_model_fix = []
    failed       = []

    if model_ok and data_ok:
        prog = st.progress(0, text="Running predictions…")
        for i, fix in enumerate(fixtures):
            try:
                pred = predict_one_match_live(
                    df, fix["home_team"], fix["away_team"],
                    fix["league_key"], today
                )
                if pred:
                    pred.update({k: fix.get(k, "") for k in
                                 ("kickoff","status","matchday")})
                    predictions.append(pred)
                else:
                    no_model_fix.append(fix)
            except Exception as e:
                failed.append(f"{fix['home_team']} vs {fix['away_team']}: {e}")
            prog.progress((i+1)/len(fixtures),
                          text=f"Predicting {i+1}/{len(fixtures)}…")
        prog.empty()
    else:
        no_model_fix = fixtures
        if not model_ok:
            st.warning("⚠️ No trained model. Run `python src/update_data.py` to train.")

    if failed:
        with st.expander(f"⚠️ {len(failed)} prediction failures (click to debug)"):
            for f in failed:
                st.markdown(f"- `{f}`")

    if not predictions and data_ok and model_ok:
        with st.expander("🔍 Debug: why no predictions?", expanded=True):
            st.markdown(f"**Fixtures fetched:** {len(fixtures)}")
            assert df is not None
            leagues_in_data = sorted(df['league_key'].unique().tolist())
            st.markdown(f"**Data rows:** {len(df):,}  |  **Leagues in data:** {leagues_in_data}")
            fixture_leagues = sorted({f['league_key'] for f in fixtures})
            missing = [l for l in fixture_leagues if l not in leagues_in_data]
            if missing:
                st.warning(
                    f"⚠️ Fixture leagues **{missing}** have no data — predictions will use "
                    f"cross-league team stats (may be less accurate). "
                    f"Run `python src/update_data.py` to pull the missing leagues."
                )
            st.markdown(f"**Model features (first 5):** {feat_mr[:5] if feat_mr else 'None'}...")
            if fixtures:
                fix0 = fixtures[0]
                st.markdown(f"**Sample fixture:** `{fix0['home_team']}` vs `{fix0['away_team']}` (league: `{fix0['league_key']}`)")
                all_teams = set(df["home_team"].tolist()) | set(df["away_team"].tolist())
                ht_match = fuzzy_team(fix0["home_team"], list(all_teams))
                at_match = fuzzy_team(fix0["away_team"], list(all_teams))
                st.markdown(f"**Home match in data:** `{ht_match}`  |  **Away match in data:** `{at_match}`")
                if not ht_match or not at_match:
                    missing = []
                    if not ht_match:
                        missing.append(fix0['home_team'])
                    if not at_match:
                        missing.append(fix0['away_team'])
                    st.error(
                        f"Teams not found in feature data: {', '.join(missing)}. "
                        f"Run `python src/update_data.py` to refresh data."
                    )
                    st.markdown("**Teams in data (sample):** " + ", ".join(sorted(list(all_teams))[:20]))
                else:
                    st.info(f"Teams resolved OK ({ht_match}, {at_match}) — check the timezone or feature error above.")
            st.markdown("**No-model fixtures:** " + str(len(no_model_fix)))

    # ── Summary metrics ───────────────────────────────────────────────────
    total_fix = len(fixtures)
    n_pred    = len(predictions)
    n_high    = sum(1 for p in predictions if p["confidence"] == "HIGH")
    n_med     = sum(1 for p in predictions if p["confidence"] == "MEDIUM")

    c1, c2, c3, c4 = st.columns(4)
    for col_s, val, lbl, color in [
        (c1, total_fix, "Fixtures Today",    "#5bc8f5"),
        (c2, n_pred,    "Predictions",        "#5bc8f5"),
        (c3, n_high,    "🟢 High Confidence", "#3de880"),
        (c4, n_med,     "🟡 Medium",          "#f5c542"),
    ]:
        col_s.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-value" style="color:{color};">{val}</div>'
            f'<div class="metric-label">{lbl}</div></div>',
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)

    high_preds = sorted(
        [p for p in predictions if p["confidence"] == "HIGH"],
        key=lambda x: -x["top_probability"]
    )
    if high_preds:
        st.markdown("### 🚨 High Confidence Picks")
        for p in high_preds:
            kick_str = (f" | {p['kickoff'][11:16]} UTC" if p.get("kickoff") else "")
            k = quick_kelly_ui(p["top_probability"], 2.0, bankroll)
            extra = ""
            if p.get("btts_prob"):
                extra += f" &nbsp;|&nbsp; BTTS: {p['btts_prob']:.0%}"
            if p.get("over_25_prob"):
                extra += f" &nbsp;|&nbsp; O2.5: {p['over_25_prob']:.0%}"
            st.markdown(f"""
            <div class="value-alert">
              <div class="value-alert-title">
                ⚡ {p['home_team']} vs {p['away_team']}
                ({LEAGUES.get(p['league_key'],{}).get('name',p['league_key'])}){kick_str}
              </div>
              <div class="value-alert-body">
                Pick: <strong>{p['top_outcome']}</strong>
                @ {p['top_probability']:.1%}{extra}
                &nbsp;|&nbsp; Kelly: <strong>{k['stake_units']:.0f}u</strong>
                &nbsp;|&nbsp; EV: {k['ev']:+.3f}
              </div>
            </div>""", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

    if predictions:
        conf_ord = {"HIGH":0,"MEDIUM":1,"LOW":2}
        predictions.sort(key=lambda x: (conf_ord.get(x["confidence"],2),
                                        -x["top_probability"]))
        st.markdown(f"### 📋 All {len(predictions)} Predictions")
        for pred in predictions:
            kick     = pred.get("kickoff","")
            kick_str = kick[11:16]+" UTC" if kick else ""
            md_str   = (f"Matchday {pred['matchday']}" if pred.get("matchday") else "")
            extra    = ""
            if pred.get("btts_prob"):
                extra += f'<span style="font-size:.78rem;color:#7ae8a0;margin-left:10px;">' \
                         f'BTTS {pred["btts_prob"]:.0%}</span>'
            if pred.get("over_25_prob"):
                extra += f'<span style="font-size:.78rem;color:#a0c8f0;margin-left:10px;">' \
                         f'O2.5 {pred["over_25_prob"]:.0%}</span>'
            st.markdown(f"""
            <div class="match-card">
              <div style="display:flex;justify-content:space-between;align-items:flex-start;">
                <div>
                  <div class="match-title">{pred['home_team']}
                    <span style="color:#5bc8f5;">vs</span> {pred['away_team']}</div>
                  <div class="match-meta">
                    {LEAGUES.get(pred['league_key'],{}).get('name',pred['league_key'])}
                    {f'• {md_str}' if md_str else ''} {f'• {kick_str}' if kick_str else ''}
                  </div>
                </div>
                <div style="text-align:right;">{badge(pred['confidence'])}</div>
              </div>
              <div style="margin-top:12px;">
                {prob_bars(pred['p_home'],pred['p_draw'],pred['p_away'])}
              </div>
              <div style="margin-top:8px;font-size:.8rem;color:#4a6280;">
                Top pick: <strong style="color:#dde4f0;">{pred['top_outcome']}</strong>
                ({pred['top_probability']:.1%}){extra}
              </div>
            </div>""", unsafe_allow_html=True)

    if show_no_model and no_model_fix:
        with st.expander(f"📅 {len(no_model_fix)} fixtures (no model prediction)"):
            for fix in no_model_fix:
                kick = fix.get("kickoff","")
                kick_str = kick[11:16]+" UTC" if kick else ""
                st.markdown(
                    f"**{fix['home_team']} vs {fix['away_team']}** — "
                    f"{LEAGUES.get(fix['league_key'],{}).get('name',fix['league_key'])}"
                    f"{f' | {kick_str}' if kick_str else ''}",
                    unsafe_allow_html=False
                )


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — ANALYST PREVIEW
# ═══════════════════════════════════════════════════════════════════════════════

elif "Analyst" in page:
    today = datetime.datetime.now(datetime.timezone.utc).date().isoformat()
    st.markdown('<div class="section-header">🧠 Analyst Preview</div>',
                unsafe_allow_html=True)
    st.markdown('<p style="color:#4a6280;">Select any fixture → get a full betting preview '
                'generated by the Oracle Analyst (Groq LLM + your ML model data).</p>',
                unsafe_allow_html=True)

    if not api_key:
        show_api_setup("FOOTBALL_DATA_KEY")
        st.stop()

    if not groq_key:
        show_api_setup("GROQ_API_KEY")
        st.stop()

    col_lf, col_ref = st.columns([4, 1])
    league_sel = col_lf.selectbox(
        "League", ["All Leagues"] + list(FD_ORG_COMP.keys()),
        format_func=lambda x: x if x == "All Leagues"
                               else LEAGUES.get(x, {}).get("name", x)
    )
    if col_ref.button("🔄", width='stretch', help="Refresh fixtures"):
        st.cache_data.clear()

    lk_filter = None if league_sel == "All Leagues" else league_sel

    with st.spinner("Fetching today's fixtures…"):
        fixtures, _ = fetch_fixtures_today(api_key, lk_filter)

    if not fixtures:
        st.info(f"No fixtures today ({today}) for selected league(s).")
        st.stop()

    fix_options = {
        f"{f['home_team']} vs {f['away_team']} ({LEAGUES.get(f['league_key'],{}).get('name',f['league_key'])})": f
        for f in fixtures
    }
    chosen_label = st.selectbox("Select a match", list(fix_options.keys()))
    chosen_fix   = fix_options[chosen_label]

    mode = st.radio(
        "Preview depth",
        ["🚀 Quick Preview (Analyst only)",
         "🔬 Full Preview (ML model + Analyst)"],
        horizontal=True,
    )

    with st.expander("📝 Add extra context for the Analyst (optional)"):
        extra_context = st.text_area(
            "e.g. key injuries, rivalry context, weather, motivation notes…",
            placeholder="Example: Arsenal missing Saka (hamstring). Chelsea on 4-match winning run...",
            height=100,
        )

    generate_btn = st.button("🧠 Generate Analyst Preview", type="primary",
                             width='stretch')

    if generate_btn:
        model_preds: dict = {}

        if "Full Preview" in mode and model_ok and data_ok:
            with st.spinner("Running ML models…"):
                try:
                    pred = predict_one_match_live(
                        df, chosen_fix["home_team"], chosen_fix["away_team"],
                        chosen_fix["league_key"], today
                    )
                    if pred:
                        model_preds = {
                            "home_win_prob":        pred["p_home"],
                            "draw_prob":            pred["p_draw"],
                            "away_win_prob":        pred["p_away"],
                            "top_outcome":          pred["top_outcome"],
                            "top_probability":      pred["top_probability"],
                            "confidence":           pred["confidence"],
                        }
                        if pred.get("btts_prob"):
                            model_preds["btts_prob"] = pred["btts_prob"]
                        if pred.get("over_25_prob"):
                            model_preds["over_25_prob"] = pred["over_25_prob"]
                        if pred.get("expected_home_goals"):
                            model_preds["expected_home_goals"] = pred["expected_home_goals"]
                            model_preds["expected_away_goals"] = pred.get("expected_away_goals", 1.1)
                            model_preds["expected_total_goals"] = (
                                model_preds["expected_home_goals"] +
                                model_preds["expected_away_goals"]
                            )
                except Exception as e:
                    st.warning(f"ML prediction failed: {e} — proceeding with analyst only.")

        elif "Full Preview" in mode and (not model_ok or not data_ok):
            st.info("ℹ️ Model or data not found — generating analyst-only preview.")

        if extra_context.strip():
            model_preds["analyst_notes"] = extra_context.strip()

        kick = chosen_fix.get("kickoff", "")
        match_data = {
            "home_team": chosen_fix["home_team"],
            "away_team": chosen_fix["away_team"],
            "league":    LEAGUES.get(chosen_fix["league_key"], {}).get("name",
                         chosen_fix["league_key"]),
            "date":      f"{today} {kick[11:16]+' UTC' if kick else ''}".strip(),
            "status":    chosen_fix.get("status", "Unknown"),
            "matchday":  chosen_fix.get("matchday", "Unknown"),
            "analyst_notes": model_preds.get("analyst_notes", ""),
        }

        with st.spinner("✍️ Analyst generating preview…"):
            try:
                from src.analyst import FootballAnalyst
                analyst = FootballAnalyst(provider="groq")
                preview = analyst.generate_preview(match_data, model_preds)
            except Exception as e:
                st.error(f"Analyst error: {e}")
                st.stop()

        st.markdown(f"""
        <div class="match-card" style="margin-bottom:12px;">
          <div class="match-title" style="font-size:1.2rem;">
            {chosen_fix['home_team']} <span style="color:#5bc8f5;">vs</span>
            {chosen_fix['away_team']}
          </div>
          <div class="match-meta">
            {LEAGUES.get(chosen_fix['league_key'],{}).get('name',chosen_fix['league_key'])}
            {'&nbsp;|&nbsp;' + kick[11:16] + ' UTC' if kick else ''}
          </div>
        </div>""", unsafe_allow_html=True)

        if model_preds and "home_win_prob" in model_preds:
            gc1, gc2, gc3 = st.columns(3)
            for col_s, lbl, val, color in [
                (gc1, "Home Win", model_preds.get("home_win_prob", 0), "#5bc8f5"),
                (gc2, "Draw",     model_preds.get("draw_prob",     0), "#f5c542"),
                (gc3, "Away Win", model_preds.get("away_win_prob", 0), "#f07070"),
            ]:
                col_s.markdown(
                    f'<div class="metric-card">'
                    f'<div class="metric-value" style="color:{color};">{val:.0%}</div>'
                    f'<div class="metric-label">{lbl}</div></div>',
                    unsafe_allow_html=True
                )

        formatted = preview
        for heading in ["Match Header", "Detailed Context", "Tactical",
                        "Supporting Trends", "Expected Outcome", "FINAL VERDICT"]:
            formatted = formatted.replace(f"**{heading}", f"**{heading}")

        st.markdown(f'<div class="analyst-card">{formatted}</div>',
                    unsafe_allow_html=True)

        col_a, col_b = st.columns(2)
        col_a.download_button(
            "📥 Download Preview",
            data=f"{chosen_fix['home_team']} vs {chosen_fix['away_team']}\n\n{preview}",
            file_name=f"preview_{chosen_fix['home_team'].replace(' ','_')}_"
                      f"{chosen_fix['away_team'].replace(' ','_')}.txt",
            mime="text/plain",
        )


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — MATCH DEEP DIVE
# ═══════════════════════════════════════════════════════════════════════════════

elif "Deep Dive" in page:
    st.markdown('<div class="section-header">🔍 Match Deep Dive + Bet Builder</div>',
                unsafe_allow_html=True)

    if not model_ok:
        st.error("No model loaded. Run `python src/update_data.py` first.")
        st.stop()
    if not data_ok:
        st.error("No feature data found. Run `python src/update_data.py` first.")
        st.stop()

    # After the guards above, both df and feat_mr are guaranteed non-None.
    assert df is not None
    assert feat_mr is not None
    assert model_mr is not None

    col1, col2, col3 = st.columns(3)
    league_key = col1.selectbox(
        "League", list(LEAGUES.keys()),
        format_func=lambda x: LEAGUES[x]["name"]
    )
    league_df  = df[df["league_key"] == league_key]
    all_teams  = sorted(
        set(league_df["home_team"].tolist()) | set(league_df["away_team"].tolist())
    )
    home_team  = col2.selectbox("Home Team", all_teams)
    away_team  = col3.selectbox("Away Team",
                                [t for t in all_teams if t != home_team])

    with st.expander("📊 Enter bookmaker odds (optional — defaults used if blank)"):
        oc1, oc2, oc3 = st.columns(3)
        odds_home = oc1.number_input("Home Win", value=2.50, min_value=1.01, step=0.05)
        odds_draw = oc2.number_input("Draw",     value=3.30, min_value=1.01, step=0.05)
        odds_away = oc3.number_input("Away Win", value=2.90, min_value=1.01, step=0.05)
        oc4, oc5 = st.columns(2)
        odds_btts_yes = oc4.number_input("BTTS Yes",   value=1.85, min_value=1.01, step=0.05)
        odds_btts_no  = oc5.number_input("BTTS No",    value=1.90, min_value=1.01, step=0.05)
        oc6, oc7 = st.columns(2)
        odds_o25  = oc6.number_input("Over 2.5",    value=1.85, min_value=1.01, step=0.05)
        odds_u25  = oc7.number_input("Under 2.5",   value=1.95, min_value=1.01, step=0.05)
        oc8, oc9 = st.columns(2)
        odds_co   = oc8.number_input("Corners O9.5",value=1.83, min_value=1.01, step=0.05)
        odds_cu   = oc9.number_input("Corners U9.5",value=1.97, min_value=1.01, step=0.05)

    market_odds = {
        "match_result": {"Home Win": odds_home, "Draw": odds_draw, "Away Win": odds_away},
        "btts":         {"Yes": odds_btts_yes, "No": odds_btts_no},
        "over_under":   {"Over 2.5": odds_o25, "Under 2.5": odds_u25},
        "corners":      {"Over 9.5": odds_co,  "Under 9.5": odds_cu},
    }

    analyse_btn = st.button("🔍 Analyse + Build Bets", type="primary",
                            width='stretch')

    if analyse_btn:
        with st.spinner("Running all market models…"):
            try:
                from src.predict import predict_all_markets
                all_markets = predict_all_markets(
                    home_team, away_team, league_key,
                    match_date=datetime.datetime.now(datetime.timezone.utc).date().isoformat(),
                    features_df=df,
                )
            except Exception as e:
                st.error(f"Prediction failed: {e}")
                st.stop()

        mr   = all_markets.get("markets", {}).get("match_result", {})
        conf = all_markets.get("overall_confidence", "LOW")

        st.markdown(f"""
        <div class="match-card" style="margin-top:16px;">
          <div style="display:flex;justify-content:space-between;">
            <div>
              <div class="match-title" style="font-size:1.3rem;">
                {home_team} vs {away_team}</div>
              <div class="match-meta">{LEAGUES[league_key]['name']}</div>
            </div>
            {badge(conf)}
          </div>
          <div style="margin-top:12px;font-size:.95rem;color:#7a90a8;">
            Model verdict:
            <strong style="color:#5bc8f5;">{all_markets['model_verdict']}</strong>
          </div>
        </div>""", unsafe_allow_html=True)

        if mr:
            st.markdown("### Outcome Probabilities")
            gc1, gc2, gc3 = st.columns(3)
            probs_mr = mr.get("probabilities", {})
            for col_s, (label, key, color) in zip(
                [gc1, gc2, gc3],
                [("Home Win","Home Win","#5bc8f5"),
                 ("Draw",    "Draw",    "#f5c542"),
                 ("Away Win","Away Win","#f07070")]
            ):
                prob = probs_mr.get(key, 0)
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=round(prob*100,1),
                    number={"suffix":"%","font":{"size":34,"color":color}},
                    gauge={
                        "axis":  {"range":[0,100],"tickcolor":"#2a4060"},
                        "bar":   {"color":color,"thickness":.28},
                        "bgcolor":"#0a0f1c","bordercolor":"#1a2a40",
                        "steps":[{"range":[0,100],"color":"#0a0f1c"}],
                    },
                    title={"text":label,"font":{"color":"#7a90a8","size":13}},
                ))
                fig.update_layout(height=190,paper_bgcolor="#07090f",
                                  font_color="#dde4f0",
                                  margin=dict(t=40,b=10,l=20,r=20))
                col_s.plotly_chart(fig, width='stretch')

        st.markdown("---")

        st.markdown("## 🏗️ Bet Builder")
        from src.betting_engine import BetBuilder
        builder = BetBuilder(bankroll=bankroll)
        result  = builder.build(
            home_team=home_team, away_team=away_team, league_key=league_key,
            predictions={
                mk: mdata["probabilities"]
                for mk, mdata in all_markets.get("markets", {}).items()
            },
            market_odds=market_odds,
        )

        mkt_order  = ["match_result","btts","over_under","corners"]
        mkt_labels = {
            "match_result": "Match Result",
            "btts":         "Both Teams to Score",
            "over_under":   "Over / Under 2.5",
            "corners":      "Corners O/U 9.5",
        }
        st.markdown('<div class="bet-builder-card"><div class="bet-builder-title">'
                    '📋 All Markets — Model Predictions</div>', unsafe_allow_html=True)
        for mk in mkt_order:
            mdata = all_markets.get("markets", {}).get(mk)
            if not mdata: continue
            top_out  = mdata["top_outcome"]
            conf_mk  = mdata["confidence"]
            probs    = mdata["probabilities"]
            label    = mkt_labels.get(mk, mk)
            st.markdown(f"""
            <div class="market-row">
              <div class="market-name">{label}</div>
              <div style="flex:1;">{prob_bars_generic(probs)}</div>
              <div style="text-align:right;min-width:120px;">
                <div class="market-pick">{top_out}</div>
                <div style="margin-top:3px;">{badge(conf_mk)}</div>
              </div>
            </div>""", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("### 💰 Value Bets")
        if result.value_bets:
            for vb in result.value_bets:
                ev_col = "#3de880" if vb.ev > 0 else "#f07070"
                st.markdown(f"""
                <div class="value-alert">
                  <div class="value-alert-title">
                    ✅ {vb.market} — <strong>{vb.outcome}</strong>
                    &nbsp;|&nbsp; Odds: {vb.odds}
                    &nbsp;|&nbsp; <span style="color:#3de880;">Edge: {vb.edge:+.1%}</span>
                  </div>
                  <div class="value-alert-body">
                    Model: {vb.model_prob:.1%} &nbsp;|&nbsp;
                    Fair: {vb.fair_prob:.1%} &nbsp;|&nbsp;
                    EV: <span style="color:{ev_col};">{vb.ev:+.3f}</span>
                    &nbsp;|&nbsp; Kelly: <strong>{vb.stake_units:.1f}u ({vb.kelly_frac:.1%})</strong>
                  </div>
                </div>""", unsafe_allow_html=True)
        else:
            st.info("No value bets detected. Try updating the odds above.")

        bs = result.best_single
        if bs:
            ev_col  = "#3de880" if bs.ev > 0 else "#f5c542"
            val_tag = "✅ VALUE BET" if bs.is_value else "⚠️ No edge"
            st.markdown("### 🎯 Best Single Bet")
            st.markdown(f"""
            <div class="bet-builder-card"
                 style="border-color:{'#0f6b25' if bs.is_value else '#1a2a40'};">
              <div class="bet-builder-title">
                {bs.market} → <span style="color:#dde4f0;">{bs.outcome}</span>
                &nbsp;&nbsp;<span style="font-size:.78rem;color:
                {'#3de880' if bs.is_value else '#f5c542'};">{val_tag}</span>
              </div>
              <div style="display:flex;gap:28px;flex-wrap:wrap;margin-top:4px;">
                <div><div class="metric-label">Odds</div>
                     <div style="font-family:'DM Mono';color:#dde4f0;
                                 font-size:1.4rem;font-weight:700;">{bs.odds}</div></div>
                <div><div class="metric-label">Model Prob</div>
                     <div style="font-family:'DM Mono';color:#5bc8f5;
                                 font-size:1.4rem;font-weight:700;">{bs.model_prob:.1%}</div></div>
                <div><div class="metric-label">EV</div>
                     <div style="font-family:'DM Mono';color:{ev_col};
                                 font-size:1.4rem;font-weight:700;">{bs.ev:+.3f}</div></div>
                <div><div class="metric-label">Kelly Stake</div>
                     <div style="font-family:'DM Mono';color:#3de880;
                                 font-size:1.4rem;font-weight:700;">{bs.stake_units:.1f}u</div></div>
              </div>
            </div>""", unsafe_allow_html=True)

        if result.best_double or result.best_treble:
            st.markdown("### 🔗 Accumulators")
            ac1, ac2 = st.columns(2)
            for col_a, acca, title in [
                (ac1, result.best_double, "⚡ Best Double"),
                (ac2, result.best_treble, "🔥 Best Treble"),
            ]:
                if acca:
                    legs_html = "".join([
                        f'<div style="font-size:.8rem;color:#7ae8a0;padding:2px 0;">'
                        f'✓ {l["market"]}: <strong>{l["outcome"]}</strong> @ {l["odds"]}'
                        f'</div>' for l in acca.as_dict()["legs"]
                    ])
                    ev_c = "#3de880" if acca.combined_ev > 0 else "#f07070"
                    with col_a:
                        st.markdown(f"""
                        <div class="acca-card">
                          <div class="acca-title">{title}</div>
                          {legs_html}
                          <div style="margin-top:12px;display:flex;gap:24px;">
                            <div><div class="metric-label">Odds</div>
                                 <div class="acca-odds">{acca.combined_odds:.2f}</div></div>
                            <div><div class="metric-label">Prob</div>
                                 <div style="font-family:'DM Mono';color:#5bc8f5;
                                             font-size:1.15rem;">{acca.combined_prob:.1%}</div></div>
                            <div><div class="metric-label">EV</div>
                                 <div style="font-family:'DM Mono';color:{ev_c};
                                             font-size:1.15rem;">{acca.combined_ev:+.3f}</div></div>
                          </div>
                          <div style="margin-top:8px;font-size:.83rem;color:#3de880;">
                            Kelly: <strong>{acca.stake_units:.1f}u ({acca.kelly_frac:.1%})</strong>
                          </div>
                        </div>""", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### 📅 Head-to-Head History")
        ht_m = fuzzy_team(home_team, df["home_team"].unique().tolist())
        at_m = fuzzy_team(away_team, df["away_team"].unique().tolist())
        h2h  = df[
            ((df["home_team"] == ht_m) & (df["away_team"] == at_m)) |
            ((df["home_team"] == at_m) & (df["away_team"] == ht_m))
        ].sort_values("date", ascending=False).head(10)
        if not h2h.empty:
            disp = h2h[["date","home_team","away_team",
                        "home_goals","away_goals","result"]].copy()
            disp["date"] = disp["date"].dt.strftime("%d %b %Y")
            disp.columns = ["Date","Home","Away","HG","AG","Result"]
            st.dataframe(disp, width='stretch', hide_index=True)
        else:
            st.info("No H2H history found.")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — BANKROLL SIMULATOR
# ═══════════════════════════════════════════════════════════════════════════════

elif "Bankroll" in page:
    st.markdown('<div class="section-header">💰 Bankroll Simulator</div>',
                unsafe_allow_html=True)

    sc1, sc2, sc3 = st.columns(3)
    win_rate = sc1.slider("Win rate",  .30, .65, .47, .01, format="%.0f%%")
    avg_odds = sc2.slider("Avg odds",  1.50, 3.50, 2.10, .05)
    n_bets   = sc3.slider("Bets",      50, 500, 200, 10)

    np.random.seed(42)
    kelly_bk, flat_bk = [bankroll], [bankroll]
    kbr, fbr = bankroll, bankroll
    for _ in range(n_bets):
        won = np.random.random() < win_rate
        p   = win_rate; b = avg_odds - 1
        fk  = max(0, (b*p - (1-p))/b) * KELLY.FRACTION
        fk  = min(fk, KELLY.MAX_STAKE_PCT)
        ks  = fk * kbr
        kbr = max(.01, kbr + (ks*(avg_odds-1) if won else -ks))
        kelly_bk.append(round(kbr, 2))
        fs  = .02 * fbr
        fbr = max(.01, fbr + (fs*(avg_odds-1) if won else -fs))
        flat_bk.append(round(fbr, 2))

    fig = go.Figure()
    fig.add_trace(go.Scatter(y=kelly_bk, name=f"Kelly 25% — {kelly_bk[-1]:.0f}u",
                             line=dict(color="#5bc8f5",width=2.5),
                             fill="tozeroy",fillcolor="rgba(91,200,245,.06)"))
    fig.add_trace(go.Scatter(y=flat_bk, name=f"Flat 2% — {flat_bk[-1]:.0f}u",
                             line=dict(color="#f5c542",width=2,dash="dash")))
    fig.add_hline(y=bankroll, line_dash="dot", line_color="#2a4060",
                  annotation_text="Start")
    fig.update_layout(paper_bgcolor="#07090f", plot_bgcolor="#0a0d17",
                      font_color="#7a90a8", height=370,
                      legend=dict(bgcolor="#0a0d17",bordercolor="#1a2235"),
                      xaxis=dict(title="Bet #",gridcolor="#111c30",color="#4a6280"),
                      yaxis=dict(title="Bankroll",gridcolor="#111c30",color="#4a6280"),
                      margin=dict(t=20,b=40,l=60,r=20))
    st.plotly_chart(fig, width='stretch')

    roi_k = (kelly_bk[-1]-bankroll)/bankroll
    roi_f = (flat_bk[-1]-bankroll)/bankroll
    bh    = np.array(kelly_bk); pk = np.maximum.accumulate(bh)
    dd    = float(((pk-bh)/pk).max())

    m1,m2,m3,m4 = st.columns(4)
    m1.metric("Kelly final",  f"{kelly_bk[-1]:,.0f}", f"{roi_k:+.1%}")
    m2.metric("Flat final",   f"{flat_bk[-1]:,.0f}",  f"{roi_f:+.1%}")
    m3.metric("Max drawdown", f"{dd:.1%}")
    m4.metric("Total bets",   str(n_bets))

    if data_ok:
        st.markdown("---")
        st.markdown("### 📊 Historical Result Distribution")
        assert df is not None
        dl = st.selectbox("League", ["All"]+list(LEAGUES.keys()),
                          format_func=lambda x: x if x=="All" else LEAGUES[x]["name"])
        sub = df if dl=="All" else df[df["league_key"]==dl]
        counts = sub["result"].value_counts()
        fig2 = go.Figure(go.Pie(
            labels=["Home Win","Draw","Away Win"],
            values=[counts.get("H",0),counts.get("D",0),counts.get("A",0)],
            hole=.5,
            marker=dict(colors=["#5bc8f5","#f5c542","#f07070"],
                        line=dict(color="#07090f",width=2)),
        ))
        fig2.update_layout(paper_bgcolor="#07090f",font_color="#7a90a8",
                           height=280,margin=dict(t=20,b=20,l=20,r=20),
                           legend=dict(bgcolor="#0a0d17"))
        dc1,dc2 = st.columns([1,1])
        dc1.plotly_chart(fig2, width='stretch')
        with dc2:
            st.markdown("<br><br>", unsafe_allow_html=True)
            for col_n, attr, lbl in [
                ("total_goals", "Avg Goals/Match", ".2f"),
                ("btts",        "BTTS Rate",       ".1%"),
                ("over_25",     "Over 2.5 Rate",   ".1%"),
            ]:
                if col_n in sub.columns:
                    val = sub[col_n].mean()
                    st.markdown(
                        f'<div class="metric-card">'
                        f'<div class="metric-value" style="color:#5bc8f5;">'
                        f'{val:{attr[1:]}}</div>'
                        f'<div class="metric-label">{attr}</div></div>',
                        unsafe_allow_html=True
                    )


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — LEAGUE ANALYTICS
# ═══════════════════════════════════════════════════════════════════════════════

elif "League" in page:
    st.markdown('<div class="section-header">📊 League Analytics</div>',
                unsafe_allow_html=True)

    if not data_ok:
        st.error("No data. Run `python src/update_data.py` first.")
        st.stop()

    assert df is not None

    la1, la2 = st.columns(2)
    league_sel = la1.selectbox("League", list(LEAGUES.keys()),
                               format_func=lambda x: LEAGUES[x]["name"])
    season_sel = la2.selectbox("Season", sorted(df["season"].unique(), reverse=True))
    sub = df[(df["league_key"]==league_sel) & (df["season"]==season_sel)].copy()

    if sub.empty:
        st.warning("No data for this selection.")
    else:
        lc1,lc2,lc3,lc4 = st.columns(4)
        lc1.metric("Matches",    len(sub))
        lc2.metric("Avg Goals",  f"{sub['total_goals'].mean():.2f}"
                                  if "total_goals" in sub.columns else "N/A")
        lc3.metric("Home Win %", f"{(sub['result']=='H').mean():.1%}")
        lc4.metric("BTTS %",     f"{sub['btts'].mean():.1%}"
                                  if "btts" in sub.columns else "N/A")

        if "home_goals" in sub.columns:
            st.markdown("### ⚽ Goals by Team")
            teams_g = []
            for team in sorted(set(sub["home_team"]) | set(sub["away_team"])):
                h  = sub[sub["home_team"] == team]
                a  = sub[sub["away_team"] == team]
                gf = h["home_goals"].sum() + a["away_goals"].sum()
                ga = h["away_goals"].sum() + a["home_goals"].sum()
                gp = len(h) + len(a)
                if gp > 0:
                    teams_g.append({"Team":team,"GF":int(gf),"GA":int(ga),"GP":gp})
            tdf = pd.DataFrame(teams_g).sort_values("GF",ascending=False)
            fig3 = go.Figure()
            fig3.add_trace(go.Bar(x=tdf["Team"],y=tdf["GF"],name="Scored",
                                  marker_color="#5bc8f5"))
            fig3.add_trace(go.Bar(x=tdf["Team"],y=tdf["GA"],name="Conceded",
                                  marker_color="#f07070"))
            fig3.update_layout(barmode="group",paper_bgcolor="#07090f",
                               plot_bgcolor="#0a0d17",font_color="#7a90a8",
                               height=350,xaxis=dict(tickangle=-45,
                               gridcolor="#111c30",color="#4a6280"),
                               yaxis=dict(gridcolor="#111c30",color="#4a6280"),
                               legend=dict(bgcolor="#0a0d17"),
                               margin=dict(t=10,b=100,l=50,r=20))
            st.plotly_chart(fig3, width='stretch')


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 6 — MODEL PERFORMANCE
# ═══════════════════════════════════════════════════════════════════════════════

elif "Model" in page:
    st.markdown('<div class="section-header">⚙️ Model Performance</div>',
                unsafe_allow_html=True)

    if not model_ok:
        st.error("No model. Run `python src/update_data.py`")
        st.stop()

    # After guard, model_mr and feat_mr are guaranteed non-None.
    assert model_mr is not None
    assert feat_mr is not None

    mc1,mc2,mc3 = st.columns(3)
    mc1.markdown('<div class="metric-card"><div class="metric-value" '
                 'style="font-size:1.3rem;">XGBoost</div>'
                 '<div class="metric-label">Algorithm</div></div>',
                 unsafe_allow_html=True)
    mc2.markdown(f'<div class="metric-card"><div class="metric-value">'
                 f'{len(feat_mr)}</div><div class="metric-label">Features</div></div>',
                 unsafe_allow_html=True)
    if data_ok:
        assert df is not None
        mc3.markdown(f'<div class="metric-card"><div class="metric-value">'
                     f'{len(df):,}</div><div class="metric-label">Training Rows</div></div>',
                     unsafe_allow_html=True)

    meta_path = PATHS.MODELS / "model_meta.pkl"
    if meta_path.exists():
        meta = joblib.load(meta_path)
        st.markdown("### 📈 Training Summary")
        tm1,tm2,tm3,tm4 = st.columns(4)
        tm1.metric("Accuracy", f"{meta.get('accuracy',0):.1%}")
        tm2.metric("F1 Score", f"{meta.get('f1',0):.3f}")
        tm3.metric("Train size", f"{meta.get('n_train',0):,}")
        tm4.metric("Trained at", meta.get("trained_at","—"))

    st.markdown("### 🎯 Feature Importance (Top 20)")
    try:
        imp = pd.DataFrame({
            "feature":    feat_mr,
            "importance": model_mr.feature_importances_,
        }).sort_values("importance", ascending=False).head(20)

        fig5 = go.Figure(go.Bar(
            x=imp["importance"], y=imp["feature"], orientation="h",
            marker=dict(color=imp["importance"],
                        colorscale=[[0,"#0f2240"],[1,"#5bc8f5"]],showscale=False),
        ))
        fig5.update_layout(
            paper_bgcolor="#07090f", plot_bgcolor="#0a0d17",
            font_color="#7a90a8", height=550,
            yaxis=dict(autorange="reversed",gridcolor="#111c30",color="#7a90a8"),
            xaxis=dict(gridcolor="#111c30",color="#4a6280",title="Importance"),
            margin=dict(t=10,b=40,l=220,r=20),
        )
        st.plotly_chart(fig5, width='stretch')
    except Exception as e:
        st.warning(f"Feature importances unavailable: {e}")

    if data_ok:
        assert df is not None
        st.markdown("### 📊 Test Season Evaluation")
        test_df = df[df["season"].isin(SEASONS.TEST)].copy()
        if not test_df.empty:
            test_df = sanitise_cols(test_df)
            for col in feat_mr:
                if col not in test_df.columns: test_df[col] = -999.0
            X_test = test_df[feat_mr].astype(np.float32).fillna(-999)
            y_test = test_df["result"].map({"H":0,"D":1,"A":2}).dropna().astype(int)
            X_test = X_test.loc[y_test.index]

            from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
            y_pred = model_mr.predict(X_test)
            acc    = accuracy_score(y_test, y_pred)
            f1     = f1_score(y_test, y_pred, average="weighted", zero_division=0)

            ec1,ec2,ec3 = st.columns(3)
            ec1.metric("Accuracy",  f"{acc:.1%}")
            ec2.metric("F1 Score",  f"{f1:.3f}")
            ec3.metric("Test size", f"{len(y_test):,}")

            cm = confusion_matrix(y_test, y_pred)
            fig6 = go.Figure(go.Heatmap(
                z=cm, x=["Home","Draw","Away"], y=["Home","Draw","Away"],
                colorscale=[[0,"#0a0d17"],[1,"#5bc8f5"]],
                text=cm, texttemplate="%{text}", showscale=False,
            ))
            fig6.update_layout(paper_bgcolor="#07090f",plot_bgcolor="#0a0d17",
                               font_color="#7a90a8",height=300,
                               xaxis=dict(title="Predicted",color="#7a90a8"),
                               yaxis=dict(title="Actual",color="#7a90a8"),
                               margin=dict(t=10,b=60,l=80,r=20))
            st.markdown("**Confusion Matrix**")
            st.plotly_chart(fig6, width='stretch')
        else:
            st.info(f"No test data for seasons: {SEASONS.TEST}")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 7 — ALERTS & SIGNAL TRACKER
# ═══════════════════════════════════════════════════════════════════════════════

elif "Alerts" in page:
    st.markdown('<div class="section-header">🔔 Alerts & Signal Tracker</div>',
                unsafe_allow_html=True)
    st.markdown('<p style="color:#4a6280;">Track value bets, monitor probability movements, '
                'and set custom threshold alerts on today\'s fixtures.</p>',
                unsafe_allow_html=True)

    if not api_key:
        show_api_setup("FOOTBALL_DATA_KEY")
        st.stop()

    today = datetime.datetime.now(datetime.timezone.utc).date().isoformat()

    tab_sig, tab_alt, tab_hist = st.tabs([
        "📡 Live Signals",
        "🔔 My Alerts",
        "📋 Tracker Log",
    ])

    # ── Shared: fetch fixtures + run predictions ─────────────────────────
    def _get_today_predictions():
        with st.spinner("Fetching today's fixtures…"):
            fixtures, api_errors = fetch_fixtures_today(api_key)
        for err in api_errors:
            st.warning(f"⚠️ {err}")
        if not fixtures:
            st.info(f"No fixtures today ({today}).")
            return [], []

        predictions = []
        if model_ok and data_ok:
            for fix in fixtures:
                try:
                    pred = predict_one_match_live(
                        df, fix["home_team"], fix["away_team"],
                        fix["league_key"], today
                    )
                    if pred:
                        pred.update({k: fix.get(k, "") for k in
                                     ("kickoff","status","matchday")})
                        predictions.append(pred)
                except Exception:
                    pass
        return fixtures, predictions

    # ── Shared: build odds map from predictions ──────────────────────────
    def _build_odds_map(predictions):
        odds_map = {}
        for pred in predictions:
            fk = f"{pred['home_team']} vs {pred['away_team']}"
            odds_map[fk] = {
                "match_result": {
                    "Home Win": 2.50, "Draw": 3.30, "Away Win": 2.90,
                },
                "btts":     {"Yes": 1.85, "No": 1.90},
                "over_under": {"Over 2.5": 1.85, "Under 2.5": 1.95},
            }
        return odds_map

    # ═══════════════════════════════════════════════════════════════════════
    # TAB 1 — LIVE SIGNALS
    # ═══════════════════════════════════════════════════════════════════════
    with tab_sig:
        fixtures, predictions = _get_today_predictions()
        if not predictions:
            st.info("No predictions available. Ensure model and feature data are present.")
        else:
            odds_map = _build_odds_map(predictions)
            signals  = generate_signals_from_predictions(predictions, odds_map)

            col_f1, col_f2, col_f3, col_f4 = st.columns(4)
            n_sig    = len(signals)
            n_val    = sum(1 for s in signals if s["is_value"])
            n_high   = sum(1 for s in signals if s["confidence"] == "HIGH")
            n_med    = sum(1 for s in signals if s["confidence"] == "MEDIUM")
            for col_s, val, lbl, color in [
                (col_f1, n_sig,  "Total Signals",  "#5bc8f5"),
                (col_f2, n_val,  "Value Bets",      "#3de880"),
                (col_f3, n_high, "High Confidence","#3de880"),
                (col_f4, n_med,  "Medium",          "#f5c542"),
            ]:
                col_s.markdown(
                    f'<div class="metric-card">'
                    f'<div class="metric-value" style="color:{color};">{val}</div>'
                    f'<div class="metric-label">{lbl}</div></div>',
                    unsafe_allow_html=True
                )

            st.markdown("<br>", unsafe_allow_html=True)

            # Filter controls
            fc1, fc2, fc3 = st.columns(3)
            filt_conf  = fc1.multiselect(
                "Confidence", ["HIGH","MEDIUM","LOW"], default=["HIGH","MEDIUM","LOW"])
            filt_mkt   = fc2.multiselect(
                "Market", ["match_result","btts","over_under","corners"],
                default=["match_result","btts","over_under"])
            filt_value = fc3.selectbox("Value filter", ["All","Value Only","Non-Value"])

            filtered = []
            for s in signals:
                if s["confidence"] not in filt_conf:
                    continue
                if s["market"] not in filt_mkt:
                    continue
                if filt_value == "Value Only" and not s["is_value"]:
                    continue
                if filt_value == "Non-Value" and s["is_value"]:
                    continue
                filtered.append(s)

            if not filtered:
                st.info("No signals match the current filters.")
            else:
                for sig in filtered:
                    ev_col  = "#3de880" if sig["is_value"] else "#f07070"
                    val_tag = "✅ VALUE" if sig["is_value"] else "—"
                    odds_str = f"{sig['odds']:.2f}" if sig["odds"] else "—"
                    kelly_str = f"{sig['kelly_stake']:.1f}u" if sig["kelly_stake"] else "—"
                    ev_str = f"{sig['ev']:+.3f}" if sig["ev"] is not None else "—"
                    st.markdown(f"""
                    <div class="match-card">
                      <div style="display:flex;justify-content:space-between;align-items:flex-start;">
                        <div>
                          <div class="match-title">{sig['home_team']}
                            <span style="color:#5bc8f5;">vs</span> {sig['away_team']}</div>
                          <div class="match-meta">
                            {LEAGUES.get(sig['league_key'],{}).get('name',sig['league_key'])}
                            &nbsp;|&nbsp; {sig['market'].replace('_',' ').title()}
                            &nbsp;|&nbsp; Pick: <strong style="color:#dde4f0;">{sig['outcome']}</strong>
                          </div>
                        </div>
                        <div style="text-align:right;">{badge(sig['confidence'])}</div>
                      </div>
                      <div style="margin-top:10px;display:flex;gap:18px;flex-wrap:wrap;font-size:.82rem;color:#7a90a8;">
                        <div>Prob: <strong style="color:#dde4f0;">{sig['probability']:.1%}</strong></div>
                        <div>Odds: <strong style="color:#dde4f0;">{odds_str}</strong></div>
                        <div>EV: <strong style="color:{ev_col};">{ev_str}</strong></div>
                        <div>Kelly: <strong style="color:#3de880;">{kelly_str}</strong></div>
                        <div>{val_tag}</div>
                      </div>
                    </div>""", unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════════════════════════
    # TAB 2 — MY ALERTS
    # ═══════════════════════════════════════════════════════════════════════
    with tab_alt:
        tracker = load_alerts_tracker()
        saved_alerts = tracker.get("alerts", [])

        st.markdown("### ➕ Create New Alert")
        with st.form("create_alert_form"):
            ca1, ca2, ca3 = st.columns(3)
            alert_fix = ca1.selectbox(
                "Fixture",
                options=[f"{f['home_team']} vs {f['away_team']}" for f in fixtures]
                if fixtures else ["No fixtures today"],
            )
            alert_market = ca2.selectbox(
                "Market", ["match_result","btts","over_under","corners"],
                format_func=lambda x: {"match_result":"Match Result","btts":"BTTS",
                                       "over_under":"Over/Under 2.5","corners":"Corners"}.get(x,x)
            )
            alert_outcome = ca3.selectbox("Outcome", ["Home Win","Draw","Away Win","Yes","Over 2.5"])

            ac4, ac5 = st.columns(2)
            alert_condition = ac4.selectbox("Condition", ["above","below"])
            alert_threshold = ac5.number_input("Threshold (probability)", value=0.60,
                                               min_value=0.01, max_value=0.99, step=0.01,
                                               format="%.2f")
            alert_notes = st.text_input("Notes (optional)", placeholder="e.g. key injury, derby context…")
            submitted = st.form_submit_button("💾 Save Alert", width='stretch')

        if submitted:
            new_alert = {
                "id":          str(uuid.uuid4())[:8],
                "fixture_key": alert_fix,
                "league_key":  "",
                "market":      alert_market,
                "outcome":     alert_outcome,
                "condition":   alert_condition,
                "threshold":   float(alert_threshold),
                "status":      "active",
                "created_at":  datetime.datetime.now().isoformat(),
                "triggered_at":None,
                "triggered_prob": None,
                "notes":       alert_notes,
            }
            # Try to enrich with league info from fixtures
            fix_map = {f"{f['home_team']} vs {f['away_team']}": f for f in fixtures}
            matched_fix = fix_map.get(alert_fix)
            if matched_fix:
                new_alert["league_key"] = matched_fix.get("league_key", "")
            saved_alerts.append(new_alert)
            tracker["alerts"] = saved_alerts
            save_alerts_tracker(tracker)
            st.success("✅ Alert saved.")
            st.rerun()

        st.markdown("---")
        st.markdown("### 📋 Active Alerts")

        if not saved_alerts:
            st.info("No alerts yet. Create one above.")
        else:
            # Check alerts against current predictions
            _, predictions_for_alerts = _get_today_predictions()
            triggered_alerts = check_alerts(predictions_for_alerts, saved_alerts)

            # Persist triggered statuses
            if triggered_alerts:
                triggered_ids = {a["id"] for a in triggered_alerts}
                updated = False
                for a in saved_alerts:
                    if a["id"] in triggered_ids and a.get("status") == "active":
                        for ta in triggered_alerts:
                            if ta["id"] == a["id"]:
                                a["status"] = "triggered"
                                a["triggered_at"] = ta.get("triggered_at")
                                a["triggered_prob"] = ta.get("triggered_prob")
                                updated = True
                if updated:
                    tracker["alerts"] = saved_alerts
                    save_alerts_tracker(tracker)

            for a in saved_alerts:
                status_col = "#3de880" if a.get("status") == "active" else "#f5c542"
                status_lbl = "🟢 Active" if a.get("status") == "active" else "🟡 Triggered"
                triggered_prob_str = ""
                if a.get("triggered_prob") is not None:
                    triggered_prob_str = (
                        f"<br><span style='color:#3de880;'>Triggered @ {a['triggered_prob']:.1%}</span>"
                    )
                notes_str = f"<br><span style='color:#4a6280;'>{a.get('notes','')}</span>" if a.get("notes") else ""
                st.markdown(f"""
                <div class="match-card">
                  <div style="display:flex;justify-content:space-between;align-items:center;">
                    <div>
                      <div class="match-title">{a['fixture_key']}</div>
                      <div class="match-meta">
                        {a['market'].replace('_',' ').title()} — {a['outcome']}
                        &nbsp;|&nbsp; {a['condition'].title()} {a['threshold']:.0%}
                        {triggered_prob_str}{notes_str}
                      </div>
                    </div>
                    <div style="text-align:right;">
                      <div style="color:{status_col};font-size:.78rem;font-weight:600;">{status_lbl}</div>
                    </div>
                  </div>
                </div>""", unsafe_allow_html=True)

                col_d1, col_d2 = st.columns(2)
                if col_d1.button("🗑️ Dismiss", key=f"dismiss_{a['id']}", width='stretch'):
                    saved_alerts = [x for x in saved_alerts if x["id"] != a["id"]]
                    tracker["alerts"] = saved_alerts
                    save_alerts_tracker(tracker)
                    st.rerun()
                if a.get("status") == "triggered" and col_d2.button("🔄 Reset", key=f"reset_{a['id']}", width='stretch'):
                    a["status"] = "active"
                    a["triggered_at"] = None
                    a["triggered_prob"] = None
                    tracker["alerts"] = saved_alerts
                    save_alerts_tracker(tracker)
                    st.rerun()

    # ═══════════════════════════════════════════════════════════════════════
    # TAB 3 — TRACKER LOG
    # ═══════════════════════════════════════════════════════════════════════
    with tab_hist:
        tracker = load_alerts_tracker()
        log_entries = tracker.get("tracked", [])
        signals_hist = tracker.get("signals", [])

        st.markdown("### 📊 Today's Tracked Signals")
        if not signals_hist:
            st.info("Signals will appear here after predictions are generated on the Live Signals tab.")
        else:
            sig_df = pd.DataFrame(signals_hist)
            if not sig_df.empty:
                display_cols = ["date","fixture_key","market","outcome",
                                "probability","confidence","is_value","ev"]
                avail_cols = [c for c in display_cols if c in sig_df.columns]
                st.dataframe(sig_df[avail_cols], width='stretch', hide_index=True)

        st.markdown("---")
        st.markdown("### 🕒 Alert History")
        if not saved_alerts:
            st.info("No alert history yet.")
        else:
            hist_df = pd.DataFrame(saved_alerts)
            if not hist_df.empty:
                hcols = ["created_at","fixture_key","market","outcome",
                         "condition","threshold","status","triggered_prob","notes"]
                hcols = [c for c in hcols if c in hist_df.columns]
                st.dataframe(hist_df[hcols], width='stretch', hide_index=True)

        if st.button("🧹 Clear Tracker Log", width='stretch'):
            tracker["tracked"] = []
            tracker["signals"] = []
            save_alerts_tracker(tracker)
            st.success("Tracker log cleared.")
            st.rerun()