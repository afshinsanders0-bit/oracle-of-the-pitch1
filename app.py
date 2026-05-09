"""
app.py — FootballIQ Streamlit Dashboard
=========================================
UPGRADES:
    - All pages pull live fixtures from football-data.org API
    - Match Deep Dive now has a full Bet Builder:
        * All 4 market predictions shown (Match Result, BTTS, O/U, Corners)
        * Value bets detected with EV and edge displayed
        * Best single bet highlighted with Kelly stake
        * Best double and treble accumulators built automatically
        * SHAP explanation for match result prediction
    - Today's Predictions uses live API fixtures, not parquet
    - Model verdict shown as the primary recommendation

Run: streamlit run app.py
Requires: FOOTBALL_DATA_KEY in .env or environment
"""
import sys, re, os, datetime
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
import time
import plotly.graph_objects as go

from config import PATHS, LEAGUES, VALUE_BET, KELLY, SEASONS

st.set_page_config(
    page_title="FootballIQ",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');
html,body,[class*="css"]{font-family:'Space Grotesk',sans-serif;}
.main,[data-testid="stAppViewContainer"]{background:#0a0e1a;}
[data-testid="stSidebar"]{background:#0d1220;border-right:1px solid #1e2d4a;}
h1,h2,h3{color:#e8edf5!important;}
p,li,label{color:#8a9bb5!important;}
.metric-card{background:linear-gradient(135deg,#0d1a2e,#0a1525);border:1px solid #1e3050;border-radius:12px;padding:20px 24px;margin:6px 0;}
.metric-value{font-size:2rem;font-weight:700;color:#4fc3f7;font-family:'JetBrains Mono',monospace;}
.metric-label{font-size:0.75rem;color:#5a7a9e;text-transform:uppercase;letter-spacing:1.5px;margin-top:4px;}
.match-card{background:#0d1a2e;border:1px solid #1e3050;border-radius:12px;padding:16px 20px;margin:8px 0;}
.match-title{font-size:1.1rem;font-weight:600;color:#e8edf5;}
.match-meta{font-size:0.8rem;color:#5a7a9e;margin-top:2px;}
.badge-high{background:#0d3321;color:#4ade80;border:1px solid #166534;padding:3px 10px;border-radius:20px;font-size:0.75rem;font-weight:600;}
.badge-medium{background:#2d2400;color:#fbbf24;border:1px solid #854d0e;padding:3px 10px;border-radius:20px;font-size:0.75rem;font-weight:600;}
.badge-low{background:#2d0f0f;color:#f87171;border:1px solid #7f1d1d;padding:3px 10px;border-radius:20px;font-size:0.75rem;font-weight:600;}
.prob-bar-wrap{display:flex;align-items:center;gap:10px;margin:4px 0;}
.prob-label{width:90px;font-size:0.8rem;color:#8a9bb5;}
.prob-bar-bg{flex:1;background:#1a2744;border-radius:4px;height:8px;}
.prob-bar-fill{height:8px;border-radius:4px;}
.prob-pct{width:42px;text-align:right;font-size:0.8rem;font-family:'JetBrains Mono',monospace;color:#e8edf5;}
.value-alert{background:linear-gradient(135deg,#0a2d1a,#062010);border:1px solid #16a34a;border-left:4px solid #22c55e;border-radius:8px;padding:14px 18px;margin:8px 0;}
.value-alert-title{color:#4ade80;font-weight:700;font-size:0.95rem;}
.value-alert-body{color:#86efac;font-size:0.85rem;margin-top:4px;}
.bet-builder-card{background:#060f1e;border:1px solid #1e3a5c;border-radius:14px;padding:20px 24px;margin:12px 0;}
.bet-builder-title{font-size:1.1rem;font-weight:700;color:#4fc3f7;margin-bottom:14px;border-bottom:1px solid #1e3050;padding-bottom:10px;}
.market-row{display:flex;justify-content:space-between;align-items:center;padding:10px 0;border-bottom:1px solid #0d1e35;}
.market-name{font-size:0.85rem;color:#5a7a9e;width:160px;}
.market-pick{font-size:0.9rem;font-weight:600;color:#e8edf5;}
.market-prob{font-family:'JetBrains Mono',monospace;color:#4fc3f7;font-size:0.85rem;}
.acca-card{background:#0a1a0e;border:1px solid #1a4a1e;border-radius:10px;padding:16px 20px;margin:8px 0;}
.acca-title{color:#4ade80;font-weight:700;font-size:0.9rem;margin-bottom:8px;}
.acca-odds{font-family:'JetBrains Mono',monospace;font-size:1.4rem;color:#4fc3f7;font-weight:700;}
.section-header{border-bottom:1px solid #1e3050;padding-bottom:12px;margin-bottom:20px;color:#e8edf5;font-size:1.3rem;font-weight:700;}
.api-setup-box{background:#0d1a2e;border:1px solid #1e3a5c;border-left:4px solid #4fc3f7;border-radius:8px;padding:20px 24px;margin:16px 0;}
.api-setup-title{color:#4fc3f7;font-weight:700;font-size:1rem;margin-bottom:8px;}
.api-setup-step{color:#8a9bb5;font-size:0.875rem;padding:3px 0;}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════

def get_api_key() -> str | None:
    key = os.environ.get("FOOTBALL_DATA_KEY", "").strip()
    if key:
        return key
    env = Path(__file__).parent / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if line.startswith("FOOTBALL_DATA_KEY="):
                k = line.split("=", 1)[1].strip().strip('"').strip("'")
                if k:
                    return k
    return None


def show_api_setup():
    st.markdown("""
    <div class="api-setup-box">
      <div class="api-setup-title">🔑 API Key Required</div>
      <div class="api-setup-step">Live data from <strong style="color:#e8edf5;">football-data.org</strong> — free, no credit card.</div><br>
      <div class="api-setup-step"><strong style="color:#e8edf5;">Step 1</strong> → Register at <a href="https://www.football-data.org/client/register" target="_blank" style="color:#4fc3f7;">football-data.org/client/register</a></div>
      <div class="api-setup-step"><strong style="color:#e8edf5;">Step 2</strong> → Check email for your API key</div>
      <div class="api-setup-step"><strong style="color:#e8edf5;">Step 3</strong> → Run in terminal:</div>
      <pre style="background:#060c18;color:#4ade80;padding:10px 14px;border-radius:6px;margin-top:8px;font-size:0.85rem;">echo 'FOOTBALL_DATA_KEY=your_key_here' > .env</pre>
      <div class="api-setup-step" style="margin-top:8px;">Or paste below for this session only:</div>
    </div>
    """, unsafe_allow_html=True)

    temp_key = st.text_input("Paste API key (session only)", type="password",
                              placeholder="e.g. abc123def456...")
    if temp_key:
        os.environ["FOOTBALL_DATA_KEY"] = temp_key.strip()
        st.success("✓ Key set. Refreshing...")
        st.rerun()


@st.cache_data(ttl=3600)
def load_features_cached() -> pd.DataFrame | None:
    for fname in ["master_features.parquet", "features_master.parquet",
                  "features.parquet", "master_with_elo.parquet", "master.parquet"]:
        p = PATHS.PROCESSED / fname
        if p.exists():
            df = pd.read_parquet(p, engine="pyarrow")
            df["date"] = pd.to_datetime(df["date"])
            df.columns = [re.sub(r'[^a-zA-Z0-9_]', '_', str(c)) for c in df.columns]
            return df
    return None


@st.cache_resource
def load_model_cached(market: str = "match_result"):
    model_map = {
        "match_result": (PATHS.MODEL_MATCH_RESULT, PATHS.MODELS / "match_result_features.pkl"),
        "btts":         (PATHS.MODEL_BTTS,          PATHS.MODELS / "btts_features.pkl"),
        "over_under":   (PATHS.MODEL_OVER_UNDER,    PATHS.MODELS / "over_under_features.pkl"),
        "corners":      (PATHS.MODEL_CORNERS,       PATHS.MODELS / "corners_features.pkl"),
    }
    mp, fp = model_map.get(market, (None, None))
    if not mp or not mp.exists():
        return None, None
    return joblib.load(mp), (joblib.load(fp) if fp.exists() else None)


@st.cache_data(ttl=300)
def fetch_fixtures_cached(api_key: str, league_key: str | None) -> tuple[list, list]:
    from src.predict import fetch_todays_fixtures
    return fetch_todays_fixtures(api_key=api_key, league_key=league_key)


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


def fuzzy_team(name: str, pool: list[str]) -> str | None:
    if name in pool:
        return name
    lm = {t.lower(): t for t in pool}
    if name.lower() in lm:
        return lm[name.lower()]
    for t in pool:
        if name.lower() in t.lower() or t.lower() in name.lower():
            return t
    words = set(name.lower().split())
    best, bs = None, 0
    for t in pool:
        ov = len(words & set(t.lower().split()))
        if ov > bs:
            bs, best = ov, t
    return best if bs >= 1 else None


def predict_one_match(df, home, away, league_key):
    """Quick single-market prediction for Today's page."""
    model, feature_names = load_model_cached("match_result")
    if model is None or feature_names is None:
        return None

    today = datetime.date.today().isoformat()
    sub   = df[(df["league_key"] == league_key) & (df["date"] < pd.to_datetime(today))].copy()
    ht    = fuzzy_team(home, sub["home_team"].unique().tolist())
    at    = fuzzy_team(away, sub["away_team"].unique().tolist())
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
                    row[col], row[ac] = row[ac].values.copy(), row[col].values.copy()
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

    row = sanitise_cols(row)
    for col in feature_names:
        if col not in row.columns:
            row[col] = -999.0
    X     = row[feature_names].astype(np.float32).fillna(-999)
    proba = model.predict_proba(X)[0]

    labels  = {0: "Home Win", 1: "Draw", 2: "Away Win"}
    top_idx = int(np.argmax(proba))
    top_p   = float(proba[top_idx])

    return {
        "home_team":      ht, "away_team": at, "league_key": league_key,
        "p_home":         float(proba[0]),
        "p_draw":         float(proba[1]),
        "p_away":         float(proba[2]),
        "top_outcome":    labels[top_idx],
        "top_probability":round(top_p, 4),
        "confidence":     ("HIGH" if top_p >= 0.65 else "MEDIUM" if top_p >= 0.50 else "LOW"),
    }


def badge(conf: str) -> str:
    cls  = {"HIGH": "badge-high", "MEDIUM": "badge-medium", "LOW": "badge-low"}.get(conf, "badge-low")
    icon = {"HIGH": "🟢", "MEDIUM": "🟡", "LOW": "🔴"}.get(conf, "⚪")
    return f'<span class="{cls}">{icon} {conf}</span>'


def prob_bars(p_home: float, p_draw: float, p_away: float) -> str:
    html = ""
    for label, prob, color in [
        ("Home Win", p_home, "#4fc3f7"),
        ("Draw",     p_draw, "#fbbf24"),
        ("Away Win", p_away, "#f87171"),
    ]:
        w = int(prob * 100)
        html += f"""
        <div class="prob-bar-wrap">
          <div class="prob-label">{label}</div>
          <div class="prob-bar-bg"><div class="prob-bar-fill" style="width:{w}%;background:{color};"></div></div>
          <div class="prob-pct">{prob:.0%}</div>
        </div>"""
    return html


def prob_bars_generic(probs: dict[str, float], color: str = "#4fc3f7") -> str:
    """Render probability bars for any market (BTTS, Over/Under, etc.)."""
    html = ""
    colors = ["#4fc3f7", "#fbbf24", "#f87171", "#a78bfa"]
    for i, (label, prob) in enumerate(probs.items()):
        c = colors[i % len(colors)]
        w = int(prob * 100)
        html += f"""
        <div class="prob-bar-wrap">
          <div class="prob-label" style="width:110px;">{label}</div>
          <div class="prob-bar-bg"><div class="prob-bar-fill" style="width:{w}%;background:{c};"></div></div>
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


# ═══════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("""
    <div style="padding:20px 0 10px 0;">
      <div style="font-size:1.6rem;font-weight:800;color:#4fc3f7;letter-spacing:-1px;">⚽ FootballIQ</div>
      <div style="font-size:0.75rem;color:#5a7a9e;margin-top:2px;letter-spacing:2px;">SPORTS INVESTING PLATFORM</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    page = st.radio("Navigation", [
        "🎯  Today's Predictions",
        "🔍  Match Deep Dive",
        "💰  Bankroll Simulator",
        "📊  League Analytics",
        "⚙️  Model Performance",
    ], label_visibility="collapsed")

    st.markdown("---")
    bankroll = st.number_input("💳 Bankroll (units)", value=1000.0, step=50.0, min_value=1.0)

    st.markdown("---")
    api_key = get_api_key()
    if api_key:
        st.markdown("""
        <div style="background:#0d3321;border:1px solid #166534;border-radius:8px;padding:10px 14px;">
          <div style="color:#4ade80;font-size:0.8rem;font-weight:600;">🟢 API Connected</div>
          <div style="color:#86efac;font-size:0.75rem;margin-top:2px;">football-data.org</div>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background:#2d0f0f;border:1px solid #7f1d1d;border-radius:8px;padding:10px 14px;">
          <div style="color:#f87171;font-size:0.8rem;font-weight:600;">🔴 No API Key</div>
          <div style="color:#fca5a5;font-size:0.75rem;margin-top:2px;">Live fixtures unavailable</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div style="font-size:0.75rem;color:#3a5070;">Data: football-data.co.uk<br>Model: XGBoost<br>Staking: 25% Kelly</div>',
                unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════
# LOAD CORE DATA
# ═══════════════════════════════════════════════════════

df       = load_features_cached()
model_mr, feat_mr = load_model_cached("match_result")
api_key  = get_api_key()
data_ok  = df is not None
model_ok = model_mr is not None and feat_mr is not None

if not data_ok:
    st.error("⚠️ No data found. Run: `python src/data_loader.py`")
    st.stop()
if not model_ok:
    st.warning("⚠️ No trained model. Run: `python src/train.py`")


# ═══════════════════════════════════════════════════════
# PAGE 1 — TODAY'S PREDICTIONS (live API)
# ═══════════════════════════════════════════════════════

if "Today" in page:
    today = datetime.date.today().isoformat()
    st.markdown(f'<div class="section-header">🎯 Today\'s Predictions — {today}</div>',
                unsafe_allow_html=True)

    if not api_key:
        show_api_setup()
        st.stop()
    if not model_ok:
        st.error("No model found. Run `python src/train.py` first.")
        st.stop()

    col_f1, col_f2 = st.columns([3, 1])
    league_filter  = col_f1.selectbox(
        "Filter by league",
        ["All Leagues"] + list(LEAGUES.keys()),
        format_func=lambda x: x if x == "All Leagues" else LEAGUES.get(x, {}).get("name", x)
    )
    if col_f2.button("🔄 Refresh", use_container_width=True):
        st.cache_data.clear()

    lk_filter = None if league_filter == "All Leagues" else league_filter

    with st.spinner("Fetching today's fixtures from football-data.org..."):
        fixtures, api_errors = fetch_fixtures_cached(api_key, lk_filter)

    for err in api_errors:
        st.warning(f"⚠️ {err}")

    if not fixtures:
        st.info(f"📅 No matches scheduled today ({today}) for the selected leagues.")
        st.stop()

    # ── Run predictions ───────────────────────────────────────────────────
    predictions = []
    failed      = []
    progress    = st.progress(0, text="Running model predictions...")

    for i, fix in enumerate(fixtures):
        try:
            pred = predict_one_match(df, fix["home_team"], fix["away_team"], fix["league_key"])
            if pred:
                pred.update({
                    "kickoff":  fix.get("kickoff", ""),
                    "status":   fix.get("status", ""),
                    "matchday": fix.get("matchday", ""),
                })
                predictions.append(pred)
            else:
                failed.append(f"{fix['home_team']} vs {fix['away_team']}")
        except Exception as e:
            failed.append(f"{fix['home_team']} vs {fix['away_team']}: {e}")
        progress.progress((i + 1) / len(fixtures), text=f"Predicting {i+1}/{len(fixtures)}...")

    progress.empty()

    if failed:
        with st.expander(f"⚠️ {len(failed)} predictions failed (team name mismatch)"):
            for f in failed:
                st.markdown(f"- `{f}`")

    if not predictions:
        st.error("No predictions generated. Run `python src/update_data.py` to refresh data.")
        st.stop()

    conf_ord = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    predictions.sort(key=lambda x: (conf_ord.get(x["confidence"], 2), -x["top_probability"]))

    n_high   = sum(1 for p in predictions if p["confidence"] == "HIGH")
    n_medium = sum(1 for p in predictions if p["confidence"] == "MEDIUM")
    n_low    = sum(1 for p in predictions if p["confidence"] == "LOW")

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f'<div class="metric-card"><div class="metric-value">{len(predictions)}</div><div class="metric-label">Predictions Today</div></div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="metric-card"><div class="metric-value" style="color:#4ade80;">{n_high}</div><div class="metric-label">🟢 High Confidence</div></div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="metric-card"><div class="metric-value" style="color:#fbbf24;">{n_medium}</div><div class="metric-label">🟡 Medium</div></div>', unsafe_allow_html=True)
    c4.markdown(f'<div class="metric-card"><div class="metric-value" style="color:#f87171;">{n_low}</div><div class="metric-label">🔴 Low</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── High confidence alerts ────────────────────────────────────────────
    high_preds = [p for p in predictions if p["confidence"] == "HIGH"]
    if high_preds:
        st.markdown("### 🚨 High Confidence Picks")
        for p in high_preds:
            kick_str = p.get("kickoff", "")
            kick_str = f" | {kick_str[11:16]} UTC" if kick_str else ""
            k = quick_kelly_ui(p["top_probability"], 2.0, bankroll)
            st.markdown(f"""
            <div class="value-alert">
              <div class="value-alert-title">⚡ {p['home_team']} vs {p['away_team']} ({p['league_key']}){kick_str}</div>
              <div class="value-alert-body">
                Model: <strong>{p['top_outcome']}</strong> @ {p['top_probability']:.1%} confidence &nbsp;|&nbsp;
                Kelly stake: <strong>{k['stake_units']:.1f} units</strong> &nbsp;|&nbsp; EV: {k['ev']:+.3f}
              </div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

    # ── All predictions ───────────────────────────────────────────────────
    st.markdown(f"### 📋 All {len(predictions)} Predictions")

    for pred in predictions:
        kick     = pred.get("kickoff", "")
        kick_str = kick[11:16] + " UTC" if kick else ""
        md_str   = f"Matchday {pred['matchday']}" if pred.get("matchday") else ""

        st.markdown(f"""
        <div class="match-card">
          <div style="display:flex;justify-content:space-between;align-items:flex-start;">
            <div>
              <div class="match-title">{pred['home_team']} <span style="color:#4fc3f7;">vs</span> {pred['away_team']}</div>
              <div class="match-meta">{LEAGUES.get(pred['league_key'],{}).get('name',pred['league_key'])} {f'• {md_str}' if md_str else ''} {f'• {kick_str}' if kick_str else ''}</div>
            </div>
            <div style="text-align:right;">{badge(pred['confidence'])}</div>
          </div>
          <div style="margin-top:12px;">{prob_bars(pred['p_home'],pred['p_draw'],pred['p_away'])}</div>
          <div style="margin-top:8px;font-size:0.82rem;color:#5a7a9e;">
            Top pick: <strong style="color:#e8edf5;">{pred['top_outcome']}</strong> ({pred['top_probability']:.1%})
          </div>
        </div>
        """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════
# PAGE 2 — MATCH DEEP DIVE + BET BUILDER
# ═══════════════════════════════════════════════════════

elif "Deep Dive" in page:
    st.markdown('<div class="section-header">🔍 Match Deep Dive + Bet Builder</div>',
                unsafe_allow_html=True)

    if not model_ok:
        st.error("No model loaded. Run `python src/train.py` first.")
        st.stop()

    # ── Match selector ────────────────────────────────────────────────────
    col1, col2, col3 = st.columns(3)
    league_key = col1.selectbox("League", list(LEAGUES.keys()),
                                format_func=lambda x: LEAGUES[x]["name"])
    league_df  = df[df["league_key"] == league_key]
    all_teams  = sorted(set(league_df["home_team"].unique()) | set(league_df["away_team"].unique()))
    home_team  = col2.selectbox("Home Team", all_teams)
    away_team  = col3.selectbox("Away Team", [t for t in all_teams if t != home_team])

    # ── Optional: manual odds input ───────────────────────────────────────
    with st.expander("📊 Enter bookmaker odds (optional — for value bet detection)"):
        oc1, oc2, oc3 = st.columns(3)
        odds_home = oc1.number_input("Home Win odds", value=2.50, min_value=1.01, step=0.05)
        odds_draw = oc2.number_input("Draw odds",     value=3.30, min_value=1.01, step=0.05)
        odds_away = oc3.number_input("Away Win odds", value=2.90, min_value=1.01, step=0.05)

        oc4, oc5 = st.columns(2)
        odds_btts_yes  = oc4.number_input("BTTS Yes odds",    value=1.85, min_value=1.01, step=0.05)
        odds_btts_no   = oc5.number_input("BTTS No odds",     value=1.90, min_value=1.01, step=0.05)

        oc6, oc7 = st.columns(2)
        odds_over_25   = oc6.number_input("Over 2.5 odds",    value=1.85, min_value=1.01, step=0.05)
        odds_under_25  = oc7.number_input("Under 2.5 odds",   value=1.95, min_value=1.01, step=0.05)

        oc8, oc9 = st.columns(2)
        odds_corners_o = oc8.number_input("Corners Over 9.5", value=1.83, min_value=1.01, step=0.05)
        odds_corners_u = oc9.number_input("Corners Under 9.5",value=1.97, min_value=1.01, step=0.05)

    market_odds = {
        "match_result": {"Home Win": odds_home, "Draw": odds_draw, "Away Win": odds_away},
        "btts":         {"Yes": odds_btts_yes, "No": odds_btts_no},
        "over_under":   {"Over 2.5": odds_over_25, "Under 2.5": odds_under_25},
        "corners":      {"Over 9.5": odds_corners_o, "Under 9.5": odds_corners_u},
    }

    analyse_btn = st.button("🔍 Analyse Match + Build Bets", type="primary",
                             use_container_width=True)

    if analyse_btn:
        with st.spinner("Running all market models..."):
            try:
                from src.predict import predict_all_markets
                all_markets = predict_all_markets(
                    home_team, away_team, league_key,
                    match_date=datetime.date.today().isoformat(),
                    features_df=df,
                )
            except Exception as e:
                st.error(f"Prediction failed: {e}")
                st.info("Make sure all models are trained: `python src/train.py`")
                st.stop()

        # ── Match header ──────────────────────────────────────────────────
        mr = all_markets.get("markets", {}).get("match_result", {})
        conf = all_markets.get("overall_confidence", "LOW")

        st.markdown(f"""
        <div class="match-card" style="margin-top:16px;">
          <div style="display:flex;justify-content:space-between;align-items:center;">
            <div>
              <div class="match-title" style="font-size:1.4rem;">{home_team} vs {away_team}</div>
              <div class="match-meta">{LEAGUES[league_key]['name']}</div>
            </div>
            {badge(conf)}
          </div>
          <div style="margin-top:14px;font-size:1rem;color:#8a9bb5;">
            Model verdict: <strong style="color:#4fc3f7;">{all_markets['model_verdict']}</strong>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Probability gauges (match result) ─────────────────────────────
        if mr:
            st.markdown("### Outcome Probabilities")
            gc1, gc2, gc3 = st.columns(3)
            probs_mr = mr.get("probabilities", {})
            for col_st, (label, key, color) in zip(
                [gc1, gc2, gc3],
                [("Home Win", "Home Win", "#4fc3f7"),
                 ("Draw",     "Draw",     "#fbbf24"),
                 ("Away Win", "Away Win", "#f87171")]
            ):
                prob = probs_mr.get(key, 0)
                fig  = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=round(prob * 100, 1),
                    number={"suffix": "%", "font": {"size": 36, "color": color}},
                    gauge={
                        "axis":    {"range": [0, 100], "tickcolor": "#3a5070"},
                        "bar":     {"color": color, "thickness": 0.3},
                        "bgcolor": "#0d1a2e", "bordercolor": "#1e3050",
                        "steps":   [{"range": [0, 100], "color": "#0d1a2e"}],
                    },
                    title={"text": label, "font": {"color": "#8a9bb5", "size": 14}},
                ))
                fig.update_layout(height=200, paper_bgcolor="#0a0e1a",
                                  font_color="#e8edf5",
                                  margin=dict(t=40, b=10, l=20, r=20))
                col_st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

        # ╔══════════════════════════════════════════════════════════════════
        # ║  BET BUILDER
        # ╚══════════════════════════════════════════════════════════════════
        st.markdown("## 🏗️ Bet Builder")

        from src.betting_engine import BetBuilder, remove_margin, calculate_ev, kelly_fraction

        builder = BetBuilder(bankroll=bankroll)
        result  = builder.build(
            home_team=home_team,
            away_team=away_team,
            league_key=league_key,
            predictions={
                mk: mdata["probabilities"]
                for mk, mdata in all_markets.get("markets", {}).items()
            },
            market_odds=market_odds,
        )

        # ── All markets summary ────────────────────────────────────────────
        st.markdown('<div class="bet-builder-card">', unsafe_allow_html=True)
        st.markdown('<div class="bet-builder-title">📋 All Markets — Model Predictions</div>',
                    unsafe_allow_html=True)

        market_display_order = ["match_result", "btts", "over_under", "corners"]
        market_labels        = {
            "match_result": "Match Result",
            "btts":         "Both Teams to Score",
            "over_under":   "Over / Under 2.5",
            "corners":      "Corners O/U 9.5",
        }

        for mk in market_display_order:
            mdata = all_markets.get("markets", {}).get(mk)
            if not mdata:
                continue
            probs     = mdata["probabilities"]
            top_out   = mdata["top_outcome"]
            top_prob  = mdata["top_probability"]
            conf_mk   = mdata["confidence"]
            label     = market_labels.get(mk, mk)

            st.markdown(f"""
            <div class="market-row">
              <div class="market-name">{label}</div>
              <div style="flex:1;">{prob_bars_generic(probs)}</div>
              <div style="text-align:right;min-width:120px;">
                <div class="market-pick">{top_out}</div>
                <div style="margin-top:2px;">{badge(conf_mk)}</div>
              </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

        # ── Value bets ────────────────────────────────────────────────────
        st.markdown("### 💰 Value Bets Detected")

        if result.value_bets:
            for vb in result.value_bets:
                ev_color = "#4ade80" if vb.ev > 0 else "#f87171"
                st.markdown(f"""
                <div class="value-alert">
                  <div class="value-alert-title">
                    ✅ {vb.market} — <strong>{vb.outcome}</strong>
                    &nbsp;|&nbsp; Odds: {vb.odds} &nbsp;|&nbsp;
                    <span style="color:#4ade80;">Edge: {vb.edge:+.1%}</span>
                  </div>
                  <div class="value-alert-body">
                    Model prob: {vb.model_prob:.1%} &nbsp;|&nbsp;
                    Fair prob: {vb.fair_prob:.1%} &nbsp;|&nbsp;
                    EV: <span style="color:{ev_color};">{vb.ev:+.3f}</span> &nbsp;|&nbsp;
                    Kelly stake: <strong>{vb.stake_units:.1f} units ({vb.kelly_frac:.1%})</strong>
                  </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No value bets detected with current odds. Try updating the bookmaker odds above.")

        # ── Best single bet ───────────────────────────────────────────────
        st.markdown("### 🎯 Best Single Bet")
        bs = result.best_single
        if bs:
            ev_col  = "#4ade80" if bs.ev > 0 else "#fbbf24"
            val_tag = "✅ VALUE BET" if bs.is_value else "⚠️ No value edge"
            st.markdown(f"""
            <div class="bet-builder-card" style="border-color:{'#16a34a' if bs.is_value else '#1e3050'};">
              <div class="bet-builder-title">
                {bs.market} → <span style="color:#e8edf5;">{bs.outcome}</span>
                &nbsp;&nbsp;<span style="font-size:0.8rem;color:{'#4ade80' if bs.is_value else '#fbbf24'};">{val_tag}</span>
              </div>
              <div style="display:flex;gap:32px;flex-wrap:wrap;margin-top:4px;">
                <div><div class="metric-label">Odds</div><div style="font-family:'JetBrains Mono';color:#e8edf5;font-size:1.5rem;font-weight:700;">{bs.odds}</div></div>
                <div><div class="metric-label">Model Prob</div><div style="font-family:'JetBrains Mono';color:#4fc3f7;font-size:1.5rem;font-weight:700;">{bs.model_prob:.1%}</div></div>
                <div><div class="metric-label">Expected Value</div><div style="font-family:'JetBrains Mono';color:{ev_col};font-size:1.5rem;font-weight:700;">{bs.ev:+.3f}</div></div>
                <div><div class="metric-label">Kelly Stake</div><div style="font-family:'JetBrains Mono';color:#4ade80;font-size:1.5rem;font-weight:700;">{bs.stake_units:.1f}u</div></div>
                <div><div class="metric-label">Stake %</div><div style="font-family:'JetBrains Mono';color:#4ade80;font-size:1.5rem;font-weight:700;">{bs.kelly_frac:.1%}</div></div>
              </div>
            </div>
            """, unsafe_allow_html=True)

        # ── Accumulators ──────────────────────────────────────────────────
        if result.best_double or result.best_treble:
            st.markdown("### 🔗 Accumulators")
            ac1, ac2 = st.columns(2)

            for col_a, acca, title in [
                (ac1, result.best_double, "⚡ Best Double"),
                (ac2, result.best_treble, "🔥 Best Treble"),
            ]:
                if acca:
                    legs_html = "".join([
                        f'<div style="font-size:0.82rem;color:#86efac;padding:2px 0;">'
                        f'✓ {l["market"]}: <strong>{l["outcome"]}</strong> @ {l["odds"]}'
                        f'</div>'
                        for l in acca.as_dict()["legs"]
                    ])
                    ev_c = "#4ade80" if acca.combined_ev > 0 else "#f87171"
                    with col_a:
                        st.markdown(f"""
                        <div class="acca-card">
                          <div class="acca-title">{title}</div>
                          {legs_html}
                          <div style="margin-top:12px;display:flex;gap:24px;">
                            <div>
                              <div class="metric-label">Combined Odds</div>
                              <div class="acca-odds">{acca.combined_odds:.2f}</div>
                            </div>
                            <div>
                              <div class="metric-label">Combined Prob</div>
                              <div style="font-family:'JetBrains Mono';color:#4fc3f7;font-size:1.2rem;">{acca.combined_prob:.1%}</div>
                            </div>
                            <div>
                              <div class="metric-label">EV</div>
                              <div style="font-family:'JetBrains Mono';color:{ev_c};font-size:1.2rem;">{acca.combined_ev:+.3f}</div>
                            </div>
                          </div>
                          <div style="margin-top:8px;font-size:0.85rem;color:#4ade80;">
                            Kelly stake: <strong>{acca.stake_units:.1f} units ({acca.kelly_frac:.1%})</strong>
                          </div>
                        </div>
                        """, unsafe_allow_html=True)

        # ── SHAP explanation ──────────────────────────────────────────────
        shap_data = mr.get("shap_values", []) if mr else []
        if shap_data:
            st.markdown("---")
            st.markdown("### 🔬 Why This Prediction? (SHAP Feature Impact)")
            shap_df = pd.DataFrame(shap_data)
            colors  = ["#4ade80" if d == "↑ increases" else "#f87171"
                       for d in shap_df["direction"]]
            fig = go.Figure(go.Bar(
                x=shap_df["shap_impact"],
                y=shap_df["feature"],
                orientation="h",
                marker_color=colors,
                text=[f"{v:+.4f}" for v in shap_df["shap_impact"]],
                textposition="outside",
            ))
            fig.update_layout(
                paper_bgcolor="#0a0e1a", plot_bgcolor="#0d1a2e",
                font_color="#8a9bb5", height=300,
                xaxis=dict(title="SHAP impact", gridcolor="#1a2744", color="#5a7a9e"),
                yaxis=dict(autorange="reversed", color="#8a9bb5"),
                margin=dict(t=10, b=40, l=220, r=60),
            )
            st.plotly_chart(fig, use_container_width=True)

        # ── H2H history ───────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("### 📅 Head-to-Head History")
        ht_match = fuzzy_team(home_team, df["home_team"].unique().tolist())
        at_match = fuzzy_team(away_team, df["away_team"].unique().tolist())

        h2h = df[
            ((df["home_team"] == ht_match) & (df["away_team"] == at_match)) |
            ((df["home_team"] == at_match) & (df["away_team"] == ht_match))
        ].sort_values("date", ascending=False).head(10)

        if not h2h.empty:
            disp = h2h[["date", "home_team", "away_team", "home_goals", "away_goals", "result"]].copy()
            disp["date"] = disp["date"].dt.strftime("%d %b %Y")
            disp.columns = ["Date", "Home", "Away", "HG", "AG", "Result"]
            st.dataframe(disp, use_container_width=True, hide_index=True)

            ht   = ht_match
            hw   = sum(1 for _, r in h2h.iterrows()
                       if (r["home_team"] == ht and r["result"] == "H") or
                          (r["away_team"] == ht and r["result"] == "A"))
            dr   = (h2h["result"] == "D").sum()
            aw   = len(h2h) - hw - int(dr)
            hc1, hc2, hc3 = st.columns(3)
            hc1.metric(f"{home_team} wins", hw)
            hc2.metric("Draws", int(dr))
            hc3.metric(f"{away_team} wins", aw)
        else:
            st.info("No head-to-head history found in dataset.")

        # ── Recent form ───────────────────────────────────────────────────
        st.markdown("### 📈 Recent Form — Last 5 Matches")
        fc1, fc2 = st.columns(2)
        for col_st, team in [(fc1, ht_match), (fc2, at_match)]:
            with col_st:
                st.markdown(f"**{team}**")
                tm = df[
                    (df["home_team"] == team) | (df["away_team"] == team)
                ].sort_values("date", ascending=False).head(5)
                for _, r in tm.iterrows():
                    is_home = r["home_team"] == team
                    opp     = r["away_team"] if is_home else r["home_team"]
                    res     = r["result"]
                    won     = (res == "H" and is_home) or (res == "A" and not is_home)
                    drew    = res == "D"
                    color   = "#4ade80" if won else "#fbbf24" if drew else "#f87171"
                    lbl     = "W" if won else "D" if drew else "L"
                    hg, ag  = int(r.get("home_goals", 0)), int(r.get("away_goals", 0))
                    venue   = "H" if is_home else "A"
                    st.markdown(
                        f'<div style="font-size:0.82rem;padding:5px 0;border-bottom:1px solid #1e3050;">'
                        f'<span style="color:{color};font-weight:700;margin-right:8px;">{lbl}</span>'
                        f'<span style="color:#8a9bb5;">({venue}) vs {opp}</span>'
                        f'<span style="float:right;font-family:\'JetBrains Mono\',monospace;color:#e8edf5;">{hg}–{ag}</span>'
                        f'</div>',
                        unsafe_allow_html=True
                    )


# ═══════════════════════════════════════════════════════
# PAGE 3 — BANKROLL SIMULATOR
# ═══════════════════════════════════════════════════════

elif "Bankroll" in page:
    st.markdown('<div class="section-header">💰 Bankroll Simulator</div>', unsafe_allow_html=True)

    sc1, sc2, sc3 = st.columns(3)
    win_rate = sc1.slider("Win rate",  0.30, 0.65, 0.47, 0.01, format="%.0f%%")
    avg_odds = sc2.slider("Avg odds",  1.50, 3.50, 2.10, 0.05)
    n_bets   = sc3.slider("Bets",      50, 500, 200, 10)

    np.random.seed(42)
    kelly_bk, flat_bk = [bankroll], [bankroll]
    kbr, fbr = bankroll, bankroll
    for _ in range(n_bets):
        won = np.random.random() < win_rate
        p   = win_rate; b = avg_odds - 1
        fk  = max(0, (b * p - (1-p)) / b) * KELLY.FRACTION
        fk  = min(fk, KELLY.MAX_STAKE_PCT)
        ks  = fk * kbr
        kbr = max(0.01, kbr + (ks*(avg_odds-1) if won else -ks))
        kelly_bk.append(round(kbr, 2))
        fs  = 0.02 * fbr
        fbr = max(0.01, fbr + (fs*(avg_odds-1) if won else -fs))
        flat_bk.append(round(fbr, 2))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=kelly_bk, name=f"Kelly 25% — {kelly_bk[-1]:.0f}u",
        line=dict(color="#4fc3f7", width=2.5),
        fill="tozeroy", fillcolor="rgba(79,195,247,0.06)"
    ))
    fig.add_trace(go.Scatter(
        y=flat_bk, name=f"Flat 2% — {flat_bk[-1]:.0f}u",
        line=dict(color="#fbbf24", width=2, dash="dash")
    ))
    fig.add_hline(y=bankroll, line_dash="dot", line_color="#3a5070", annotation_text="Start")
    fig.update_layout(
        paper_bgcolor="#0a0e1a", plot_bgcolor="#0d1a2e", font_color="#8a9bb5",
        height=380, legend=dict(bgcolor="#0d1a2e", bordercolor="#1e3050"),
        xaxis=dict(title="Bet #", gridcolor="#1a2744", color="#5a7a9e"),
        yaxis=dict(title="Bankroll", gridcolor="#1a2744", color="#5a7a9e"),
        margin=dict(t=20, b=40, l=60, r=20),
    )
    st.plotly_chart(fig, use_container_width=True)

    roi_k = (kelly_bk[-1] - bankroll) / bankroll
    roi_f = (flat_bk[-1]  - bankroll) / bankroll
    bh    = np.array(kelly_bk); pk = np.maximum.accumulate(bh)
    dd    = float(((pk - bh) / pk).max())

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Kelly final",  f"{kelly_bk[-1]:,.0f}", f"{roi_k:+.1%}")
    m2.metric("Flat final",   f"{flat_bk[-1]:,.0f}",  f"{roi_f:+.1%}")
    m3.metric("Max drawdown", f"{dd:.1%}")
    m4.metric("Bets",         str(n_bets))

    st.markdown("---")
    st.markdown("### 📊 Historical Result Distribution")
    dist_league = st.selectbox("League", ["All"] + list(LEAGUES.keys()),
                               format_func=lambda x: x if x == "All" else LEAGUES[x]["name"])
    sub = df if dist_league == "All" else df[df["league_key"] == dist_league]
    counts = sub["result"].value_counts()
    fig2   = go.Figure(go.Pie(
        labels=["Home Win", "Draw", "Away Win"],
        values=[counts.get("H", 0), counts.get("D", 0), counts.get("A", 0)],
        hole=0.5,
        marker=dict(colors=["#4fc3f7", "#fbbf24", "#f87171"],
                    line=dict(color="#0a0e1a", width=2)),
    ))
    fig2.update_layout(paper_bgcolor="#0a0e1a", font_color="#8a9bb5",
                       height=300, margin=dict(t=20, b=20, l=20, r=20),
                       legend=dict(bgcolor="#0d1a2e"))
    dc1, dc2 = st.columns([1, 1])
    dc1.plotly_chart(fig2, use_container_width=True)
    with dc2:
        st.markdown("<br><br>", unsafe_allow_html=True)
        if "total_goals" in sub.columns:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{sub["total_goals"].mean():.2f}</div><div class="metric-label">Avg Goals/Match</div></div>', unsafe_allow_html=True)
        if "btts" in sub.columns:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{sub["btts"].mean():.1%}</div><div class="metric-label">BTTS Rate</div></div>', unsafe_allow_html=True)
        if "over_25" in sub.columns:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{sub["over_25"].mean():.1%}</div><div class="metric-label">Over 2.5 Rate</div></div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════
# PAGE 4 — LEAGUE ANALYTICS
# ═══════════════════════════════════════════════════════

elif "League" in page:
    st.markdown('<div class="section-header">📊 League Analytics</div>', unsafe_allow_html=True)

    la1, la2 = st.columns(2)
    league_sel = la1.selectbox("League", list(LEAGUES.keys()),
                               format_func=lambda x: LEAGUES[x]["name"])
    season_sel = la2.selectbox("Season", sorted(df["season"].unique(), reverse=True))
    sub = df[(df["league_key"] == league_sel) & (df["season"] == season_sel)].copy()

    if sub.empty:
        st.warning("No data for this selection.")
    else:
        lc1, lc2, lc3, lc4 = st.columns(4)
        lc1.metric("Matches",    len(sub))
        lc2.metric("Avg Goals",  f"{sub['total_goals'].mean():.2f}" if "total_goals" in sub.columns else "N/A")
        lc3.metric("Home Win %", f"{(sub['result']=='H').mean():.1%}")
        lc4.metric("BTTS %",     f"{sub['btts'].mean():.1%}" if "btts" in sub.columns else "N/A")

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
                    teams_g.append({"Team": team, "GF": int(gf), "GA": int(ga), "GP": gp})
            tdf = pd.DataFrame(teams_g).sort_values("GF", ascending=False)
            fig3 = go.Figure()
            fig3.add_trace(go.Bar(x=tdf["Team"], y=tdf["GF"], name="Scored",   marker_color="#4fc3f7"))
            fig3.add_trace(go.Bar(x=tdf["Team"], y=tdf["GA"], name="Conceded", marker_color="#f87171"))
            fig3.update_layout(
                barmode="group", paper_bgcolor="#0a0e1a", plot_bgcolor="#0d1a2e",
                font_color="#8a9bb5", height=360,
                xaxis=dict(tickangle=-45, gridcolor="#1a2744", color="#5a7a9e"),
                yaxis=dict(gridcolor="#1a2744", color="#5a7a9e"),
                legend=dict(bgcolor="#0d1a2e"),
                margin=dict(t=10, b=100, l=50, r=20),
            )
            st.plotly_chart(fig3, use_container_width=True)

        if "total_goals" in sub.columns:
            st.markdown("### 📈 Avg Goals/Match Over Season")
            monthly = sub.set_index("date").resample("ME")["total_goals"].mean().reset_index()
            fig4    = go.Figure(go.Scatter(
                x=monthly["date"], y=monthly["total_goals"],
                fill="tozeroy", fillcolor="rgba(79,195,247,0.08)",
                line=dict(color="#4fc3f7", width=2),
                mode="lines+markers", marker=dict(size=6, color="#4fc3f7"),
            ))
            fig4.update_layout(
                paper_bgcolor="#0a0e1a", plot_bgcolor="#0d1a2e",
                font_color="#8a9bb5", height=260,
                xaxis=dict(gridcolor="#1a2744", color="#5a7a9e"),
                yaxis=dict(gridcolor="#1a2744", color="#5a7a9e", title="Avg goals"),
                margin=dict(t=10, b=40, l=50, r=20),
            )
            st.plotly_chart(fig4, use_container_width=True)


# ═══════════════════════════════════════════════════════
# PAGE 5 — MODEL PERFORMANCE
# ═══════════════════════════════════════════════════════

elif "Model" in page:
    st.markdown('<div class="section-header">⚙️ Model Performance</div>', unsafe_allow_html=True)

    if not model_ok:
        st.error("No model. Run `python src/train.py`")
        st.stop()

    mc1, mc2, mc3 = st.columns(3)
    mc1.markdown('<div class="metric-card"><div class="metric-value">XGBoost</div><div class="metric-label">Algorithm</div></div>', unsafe_allow_html=True)
    mc2.markdown(f'<div class="metric-card"><div class="metric-value">{len(feat_mr)}</div><div class="metric-label">Features</div></div>', unsafe_allow_html=True)
    mc3.markdown(f'<div class="metric-card"><div class="metric-value">{len(df):,}</div><div class="metric-label">Training Matches</div></div>', unsafe_allow_html=True)

    st.markdown("### 🎯 Feature Importance (Top 20)")
    try:
        imp = pd.DataFrame({
            "feature":    feat_mr,
            "importance": model_mr.feature_importances_,
        }).sort_values("importance", ascending=False).head(20)

        fig5 = go.Figure(go.Bar(
            x=imp["importance"], y=imp["feature"], orientation="h",
            marker=dict(color=imp["importance"],
                        colorscale=[[0, "#1a3a5c"], [1, "#4fc3f7"]],
                        showscale=False),
        ))
        fig5.update_layout(
            paper_bgcolor="#0a0e1a", plot_bgcolor="#0d1a2e", font_color="#8a9bb5",
            height=560,
            yaxis=dict(autorange="reversed", gridcolor="#1a2744", color="#8a9bb5"),
            xaxis=dict(gridcolor="#1a2744", color="#5a7a9e", title="Importance"),
            margin=dict(t=10, b=40, l=220, r=20),
        )
        st.plotly_chart(fig5, use_container_width=True)
    except Exception as e:
        st.warning(f"Feature importances unavailable: {e}")

    st.markdown("### 📊 Test Season Evaluation")
    test_df = df[df["season"].isin(SEASONS.TEST)].copy()
    if not test_df.empty:
        test_df = sanitise_cols(test_df)
        for col in feat_mr:
            if col not in test_df.columns:
                test_df[col] = -999.0
        X_test = test_df[feat_mr].astype(np.float32).fillna(-999)
        y_test = test_df["result"].map({"H": 0, "D": 1, "A": 2}).dropna().astype(int)
        X_test = X_test.loc[y_test.index]

        from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
        y_pred = model_mr.predict(X_test)
        acc    = accuracy_score(y_test, y_pred)
        f1     = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        ec1, ec2, ec3 = st.columns(3)
        ec1.metric("Accuracy",  f"{acc:.1%}")
        ec2.metric("F1 Score",  f"{f1:.3f}")
        ec3.metric("Test size", f"{len(y_test):,}")

        cm   = confusion_matrix(y_test, y_pred)
        fig6 = go.Figure(go.Heatmap(
            z=cm, x=["Home Win", "Draw", "Away Win"],
            y=["Home Win", "Draw", "Away Win"],
            colorscale=[[0, "#0d1a2e"], [1, "#4fc3f7"]],
            text=cm, texttemplate="%{text}", showscale=False,
        ))
        fig6.update_layout(
            paper_bgcolor="#0a0e1a", plot_bgcolor="#0d1a2e",
            font_color="#8a9bb5", height=320,
            xaxis=dict(title="Predicted", color="#8a9bb5"),
            yaxis=dict(title="Actual",    color="#8a9bb5"),
            margin=dict(t=10, b=60, l=80, r=20),
        )
        st.markdown("**Confusion Matrix**")
        st.plotly_chart(fig6, use_container_width=True)
    else:
        st.info(f"No test data found for seasons: {SEASONS.TEST}")