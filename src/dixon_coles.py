"""
dixon_coles.py — Dixon-Coles Bivariate Poisson Model
======================================================
PURPOSE:
    Model the joint distribution of (home_goals, away_goals) using a
    bivariate Poisson with the Dixon-Coles (1997) low-score correction.
    Predicts: 1X2, BTTS, Over/Under, exact scorelines.

WHY BETTER THAN XGBOOST FOR GOALS:
    XGBoost treats goals as a classification problem (Over/Under 2.5 = yes/no).
    Dixon-Coles models the full *distribution* of scores — it knows that
    2-1 is far more likely than 5-4 even when both are "over 2.5".
    This gives sharper probability estimates for:
      - BTTS (both teams must score ≥ 1)
      - Over/Under (integrates the full score distribution)
      - Correct score markets (exact scorelines)

THE MODEL:
    E[home_goals] = attack_home * defence_away * home_advantage
    E[away_goals] = attack_away * defence_home

    Parameters per team: attack_i, defence_i (positive reals)
    Global parameters: home_advantage (single scalar), rho (low-score correction)

    Rho correction (Dixon-Coles Eq 3):
        τ(0,0) = 1 - λ_h * λ_a * ρ
        τ(1,0) = 1 + λ_a * ρ
        τ(0,1) = 1 + λ_h * ρ
        τ(1,1) = 1 - ρ
        τ(x,y) = 1 otherwise

    Typical ρ ≈ -0.13 (slightly increases 0-0 and 1-1 probabilities)

TIME DECAY:
    Recent matches weighted more heavily.
    weight = exp(-xi * days_since_match)
    xi = 0.005 (≈ half-weight after 138 days = ~1 season)

USAGE:
    from src.dixon_coles import DixonColesModel

    dc = DixonColesModel()
    dc.fit(master_df)                       # trains on full history
    probs = dc.predict_probs("Arsenal", "Chelsea")
    # → {p_home, p_draw, p_away, p_btts, p_over_25, p_over_15,
    #    scorelines: {(0,0): 0.05, (1,0): 0.11, ...}, lambda_home, lambda_away}

    joblib.dump(dc, PATHS.MODEL_DIXON_COLES)  # save
"""

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import joblib
from scipy.optimize import minimize
from scipy.stats import poisson
from loguru import logger

from config import PATHS, MODEL_PARAMS, SEASONS

logger.remove()
logger.add(sys.stderr, level="INFO",
           format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}")

warnings.filterwarnings("ignore", category=RuntimeWarning)


# ═══════════════════════════════════════════════════════
# SECTION 1 — TIME DECAY WEIGHTS
# ═══════════════════════════════════════════════════════

def time_decay_weights(
    dates: pd.Series,
    reference_date: pd.Timestamp | None = None,
    xi: float = 0.005,
) -> np.ndarray:
    """
    Exponential time decay: w = exp(-xi * days_ago).

    xi = 0.005 means a match 138 days ago has weight 0.5.
    Larger xi = faster decay (focus on recent form).
    Smaller xi = treat all history more equally.

    Args:
        dates:          Series of match dates
        reference_date: Date to measure from (default: today)
        xi:             Decay rate

    Returns:
        Array of weights, same length as dates.
    """
    if reference_date is None:
        reference_date = pd.Timestamp.today()
    days_ago = (reference_date - pd.to_datetime(dates)).dt.days.clip(lower=0)
    # FIX: np.asarray() narrows the type to a concrete ndarray[Any, dtype[Any]],
    # resolving Pylance reportOperatorIssue (float * ArrayLike | ExtensionArray).
    # .values alone is typed as ArrayLike | ndarray which is still too broad.
    return np.exp(-xi * np.asarray(days_ago, dtype=float))


# ═══════════════════════════════════════════════════════
# SECTION 2 — LOW-SCORE CORRECTION (tau)
# ═══════════════════════════════════════════════════════

def tau(home_goals: int, away_goals: int,
        lambda_h: float, lambda_a: float,
        rho: float) -> float:
    """
    Dixon-Coles correction factor for low-scoring matches.

    Slightly adjusts probabilities for scores (0,0), (1,0), (0,1), (1,1)
    to better fit observed goal distributions (Poisson underestimates 0-0 etc).

    Values outside {0,1} × {0,1} return 1.0 (no correction).
    """
    if home_goals == 0 and away_goals == 0:
        return 1.0 - lambda_h * lambda_a * rho
    elif home_goals == 1 and away_goals == 0:
        return 1.0 + lambda_a * rho
    elif home_goals == 0 and away_goals == 1:
        return 1.0 + lambda_h * rho
    elif home_goals == 1 and away_goals == 1:
        return 1.0 - rho
    else:
        return 1.0


def score_probability(
    home_goals: int,
    away_goals: int,
    lambda_h: float,
    lambda_a: float,
    rho: float,
) -> float:
    """
    P(score = (x, y)) under the Dixon-Coles model.

    P = τ(x,y) * Poisson(x; λ_h) * Poisson(y; λ_a)

    FIX: explicit float() wrapping ensures Pylance sees a plain float return
    rather than NDArray[float64] | float (reportReturnType false positive from
    numpy scalar arithmetic).
    """
    t  = tau(home_goals, away_goals, lambda_h, lambda_a, rho)
    ph = poisson.pmf(home_goals, lambda_h)
    pa = poisson.pmf(away_goals, lambda_a)
    # float() narrows the numpy scalar to Python float, silencing reportReturnType
    return float(max(t * ph * pa, 1e-10))


# ═══════════════════════════════════════════════════════
# SECTION 3 — NEGATIVE LOG-LIKELIHOOD
# ═══════════════════════════════════════════════════════

def _neg_log_likelihood(
    params: np.ndarray,
    teams: list[str],
    home_teams: np.ndarray,
    away_teams: np.ndarray,
    home_goals: np.ndarray,
    away_goals: np.ndarray,
    weights: np.ndarray,
    n_teams: int,
) -> float:
    """
    Compute weighted negative log-likelihood for the Dixon-Coles model.

    Parameter vector layout:
        params[0:n_teams]         = log(attack_i)    for each team
        params[n_teams:2*n_teams] = log(defence_i)   for each team
        params[2*n_teams]         = log(home_advantage)
        params[2*n_teams + 1]     = rho  (unconstrained — clipped later)
    """
    team_idx = {t: i for i, t in enumerate(teams)}

    attack  = np.exp(params[:n_teams])
    defence = np.exp(params[n_teams:2*n_teams])
    home_adv = np.exp(params[2*n_teams])
    rho      = params[2*n_teams + 1]

    nll = 0.0
    for i in range(len(home_goals)):
        hi = team_idx.get(home_teams[i])
        ai = team_idx.get(away_teams[i])
        if hi is None or ai is None:
            continue

        lam_h = attack[hi] * defence[ai] * home_adv
        lam_a = attack[ai] * defence[hi]

        # Clip lambdas for numerical stability
        lam_h = np.clip(lam_h, 1e-4, 15.0)
        lam_a = np.clip(lam_a, 1e-4, 15.0)

        p = score_probability(
            int(home_goals[i]), int(away_goals[i]),
            lam_h, lam_a, rho
        )
        nll -= weights[i] * np.log(max(p, 1e-10))

    return nll


# ═══════════════════════════════════════════════════════
# SECTION 4 — MAIN MODEL CLASS
# ═══════════════════════════════════════════════════════

class DixonColesModel:
    """
    Dixon-Coles bivariate Poisson model for football score prediction.

    Attributes set after fit():
        teams_            : sorted list of team names
        attack_           : dict {team: attack_strength}
        defence_          : dict {team: defence_weakness (lower = better defence)}
        home_advantage_   : scalar home advantage multiplier
        rho_              : low-score correction parameter
        fitted_           : True after fit() is called
        last_fit_date_    : date of most recent training match
    """

    def __init__(self, xi: float = 0.005, max_goals: int = 8):
        self.xi        = xi          # Time decay rate
        self.max_goals = max_goals   # Truncation for score matrix
        self.fitted_   = False

    # ──────────────────────────────────────────────────────────────────────
    # FIT
    # ──────────────────────────────────────────────────────────────────────

    def fit(
        self,
        df: pd.DataFrame,
        league_key: str | None = None,
        seasons: list[str] | None = None,
    ) -> "DixonColesModel":
        """
        Fit the model to historical match data.

        Args:
            df:          Master DataFrame with home_team, away_team,
                         home_goals, away_goals, date, league_key, season
            league_key:  Filter to one league (None = all)
            seasons:     Filter to specific seasons (None = all)

        Returns:
            self (for chaining)
        """
        sub = df.copy()
        if league_key:
            sub = sub[sub["league_key"] == league_key]
        if seasons:
            sub = sub[sub["season"].isin(seasons)]

        sub = sub.dropna(subset=["home_goals","away_goals","date"]).copy()
        sub = sub[sub["result"].isin(["H","D","A"])].copy()
        sub["date"] = pd.to_datetime(sub["date"])
        sub.sort_values("date", inplace=True)

        self.last_fit_date_ = sub["date"].max()
        self.league_key_    = league_key

        teams           = sorted(set(sub["home_team"]) | set(sub["away_team"]))
        self.teams_     = teams
        n_teams         = len(teams)

        home_teams  = sub["home_team"].values
        away_teams  = sub["away_team"].values
        home_goals  = sub["home_goals"].values.astype(int)
        away_goals  = sub["away_goals"].values.astype(int)
        weights     = time_decay_weights(sub["date"], xi=self.xi)

        logger.info(
            f"Fitting Dixon-Coles: {len(sub):,} matches | "
            f"{n_teams} teams | league={league_key or 'ALL'}"
        )

        # ── Initial parameters ─────────────────────────────────────────────
        # All attack/defence start at 0 (log scale → exp(0) = 1.0)
        # home_advantage starts at log(1.3) ≈ 0.26
        # rho starts at -0.1
        x0 = np.zeros(2 * n_teams + 2)
        x0[2 * n_teams]     = np.log(1.3)    # home advantage ≈ 1.3x
        x0[2 * n_teams + 1] = -0.1           # rho

        # ── Constraint: sum of log-attack = 0 (identifiability) ───────────
        # Without this, attack and defence can trade off arbitrarily.
        constraints = [{
            "type": "eq",
            "fun": lambda x: np.sum(x[:n_teams]),
        }]

        # ── Bounds for rho: (-1, 1) ────────────────────────────────────────
        bounds = [(None, None)] * (2 * n_teams + 1) + [(-0.99, 0.99)]

        result = minimize(
            _neg_log_likelihood,
            x0,
            args=(teams, home_teams, away_teams,
                  home_goals, away_goals, weights, n_teams),
            method="L-BFGS-B",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 2000, "ftol": 1e-9},
        )

        if not result.success:
            logger.warning(f"Optimisation did not fully converge: {result.message}")

        # ── Extract parameters ─────────────────────────────────────────────
        params  = result.x
        attack  = np.exp(params[:n_teams])
        defence = np.exp(params[n_teams:2*n_teams])

        self.attack_         = {t: attack[i]  for i, t in enumerate(teams)}
        self.defence_        = {t: defence[i] for i, t in enumerate(teams)}
        self.home_advantage_ = float(np.exp(params[2*n_teams]))
        self.rho_            = float(np.clip(params[2*n_teams + 1], -0.99, 0.99))
        self.fitted_         = True

        logger.info(
            f"Fitted: home_advantage={self.home_advantage_:.3f} | "
            f"rho={self.rho_:.4f}"
        )
        logger.info(
            "Top 5 attack ratings: " +
            str(dict(sorted(self.attack_.items(),
                            key=lambda x: -x[1])[:5]))
        )
        return self

    # ──────────────────────────────────────────────────────────────────────
    # PREDICT LAMBDAS
    # ──────────────────────────────────────────────────────────────────────

    def get_lambdas(
        self,
        home_team: str,
        away_team: str,
    ) -> tuple[float, float]:
        """
        Return expected goals (λ_home, λ_away) for a fixture.

        Uses average team strength for unknown teams (graceful degradation).
        """
        self._check_fitted()

        avg_attack  = float(np.mean(list(self.attack_.values())))
        avg_defence = float(np.mean(list(self.defence_.values())))

        h_attack  = self.attack_.get(home_team,  avg_attack)
        h_defence = self.defence_.get(home_team, avg_defence)
        a_attack  = self.attack_.get(away_team,  avg_attack)
        a_defence = self.defence_.get(away_team, avg_defence)

        lambda_h = h_attack * a_defence * self.home_advantage_
        lambda_a = a_attack * h_defence

        return float(np.clip(lambda_h, 0.1, 12.0)), \
               float(np.clip(lambda_a, 0.1, 12.0))

    # ──────────────────────────────────────────────────────────────────────
    # PREDICT PROBABILITIES
    # ──────────────────────────────────────────────────────────────────────

    def predict_probs(
        self,
        home_team: str,
        away_team: str,
        include_scorelines: bool = True,
    ) -> dict:
        """
        Full probability prediction for a match.

        Returns:
            {
                "home_team":    ...,
                "away_team":    ...,
                "lambda_home":  float,       # expected home goals
                "lambda_away":  float,       # expected away goals
                "p_home":       float,       # P(home win)
                "p_draw":       float,       # P(draw)
                "p_away":       float,       # P(away win)
                "p_btts":       float,       # P(both teams score ≥1)
                "p_over_15":    float,       # P(total goals > 1.5)
                "p_over_25":    float,       # P(total goals > 2.5)
                "p_over_35":    float,       # P(total goals > 3.5)
                "p_under_25":   float,       # 1 - p_over_25
                "expected_total_goals": float,
                "scorelines":   {(h,a): prob, ...}  (if include_scorelines)
            }
        """
        self._check_fitted()

        lambda_h, lambda_a = self.get_lambdas(home_team, away_team)

        # Build score matrix up to max_goals
        M = self.max_goals + 1
        score_matrix = np.zeros((M, M))

        for h in range(M):
            for a in range(M):
                score_matrix[h, a] = score_probability(
                    h, a, lambda_h, lambda_a, self.rho_
                )

        # Normalise (small numerical errors from tau correction)
        score_matrix /= score_matrix.sum()

        # ── 1X2 probabilities ──────────────────────────────────────────────
        p_home = float(np.sum(np.tril(score_matrix, -1)))   # home_goals > away_goals
        p_draw = float(np.sum(np.diag(score_matrix)))
        p_away = float(np.sum(np.triu(score_matrix, 1)))

        # Normalise to sum exactly to 1
        total  = p_home + p_draw + p_away
        p_home, p_draw, p_away = p_home/total, p_draw/total, p_away/total

        # ── BTTS ───────────────────────────────────────────────────────────
        p_btts = float(np.sum(score_matrix[1:, 1:]))

        # ── Over/Under ─────────────────────────────────────────────────────
        p_over_15 = p_over_25 = p_over_35 = 0.0
        for h in range(M):
            for a in range(M):
                total_goals = h + a
                p = score_matrix[h, a]
                if total_goals > 1.5: p_over_15 += p
                if total_goals > 2.5: p_over_25 += p
                if total_goals > 3.5: p_over_35 += p

        # ── Expected goals ─────────────────────────────────────────────────
        exp_total = lambda_h + lambda_a   # Poisson mean is λ

        result = {
            "home_team":            home_team,
            "away_team":            away_team,
            "lambda_home":          round(lambda_h, 3),
            "lambda_away":          round(lambda_a, 3),
            "p_home":               round(p_home, 4),
            "p_draw":               round(p_draw, 4),
            "p_away":               round(p_away, 4),
            "p_btts":               round(p_btts, 4),
            "p_over_15":            round(p_over_15, 4),
            "p_over_25":            round(p_over_25, 4),
            "p_over_35":            round(p_over_35, 4),
            "p_under_25":           round(1.0 - p_over_25, 4),
            "expected_total_goals": round(exp_total, 3),
        }

        if include_scorelines:
            scorelines = {}
            for h in range(M):
                for a in range(M):
                    p = score_matrix[h, a]
                    if p > 0.001:   # Only include meaningful probabilities
                        scorelines[(h, a)] = round(float(p), 4)
            # Sort by probability
            result["scorelines"] = dict(
                sorted(scorelines.items(), key=lambda x: -x[1])
            )

        return result

    # ──────────────────────────────────────────────────────────────────────
    # BATCH PREDICTIONS (for backtesting)
    # ──────────────────────────────────────────────────────────────────────

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add Dixon-Coles predictions to a DataFrame of fixtures.

        Adds columns: dc_p_home, dc_p_draw, dc_p_away, dc_p_btts,
                      dc_p_over_25, dc_lambda_home, dc_lambda_away

        Args:
            df: DataFrame with home_team, away_team columns

        Returns:
            DataFrame with dc_ prediction columns added.
        """
        self._check_fitted()
        results = []
        for _, row in df.iterrows():
            try:
                pred = self.predict_probs(
                    row["home_team"], row["away_team"],
                    include_scorelines=False
                )
                results.append({
                    "dc_lambda_home": pred["lambda_home"],
                    "dc_lambda_away": pred["lambda_away"],
                    "dc_p_home":      pred["p_home"],
                    "dc_p_draw":      pred["p_draw"],
                    "dc_p_away":      pred["p_away"],
                    "dc_p_btts":      pred["p_btts"],
                    "dc_p_over_25":   pred["p_over_25"],
                    "dc_p_over_15":   pred["p_over_15"],
                    "dc_exp_goals":   pred["expected_total_goals"],
                })
            except Exception:
                results.append({k: np.nan for k in [
                    "dc_lambda_home","dc_lambda_away","dc_p_home",
                    "dc_p_draw","dc_p_away","dc_p_btts",
                    "dc_p_over_25","dc_p_over_15","dc_exp_goals"
                ]})

        pred_df = pd.DataFrame(results)
        return pd.concat([df.reset_index(drop=True), pred_df], axis=1)

    # ──────────────────────────────────────────────────────────────────────
    # LEADERBOARD
    # ──────────────────────────────────────────────────────────────────────

    def team_strengths(self) -> pd.DataFrame:
        """
        Return a DataFrame of team attack/defence strengths, sorted by attack.
        Useful for quickly checking model sanity.
        """
        self._check_fitted()
        rows = []
        for team in self.teams_:
            rows.append({
                "team":    team,
                "attack":  round(self.attack_[team],  3),
                "defence": round(self.defence_[team], 3),
                # Expected goals at home vs average opponent
                "expected_home_goals": round(
                    self.attack_[team] *
                    np.mean(list(self.defence_.values())) *
                    self.home_advantage_, 2
                ),
            })
        return (pd.DataFrame(rows)
                  .sort_values("attack", ascending=False)
                  .reset_index(drop=True))

    # ──────────────────────────────────────────────────────────────────────
    # HELPERS
    # ──────────────────────────────────────────────────────────────────────

    def _check_fitted(self):
        if not self.fitted_:
            raise RuntimeError("Model not fitted. Call fit() first.")

    def save(self, path: Path | None = None) -> Path:
        """Save model to disk."""
        path = path or PATHS.MODEL_DIXON_COLES
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        logger.info(f"Dixon-Coles model saved: {path}")
        return path

    @classmethod
    def load(cls, path: Path | None = None) -> "DixonColesModel":
        """Load model from disk."""
        path = path or PATHS.MODEL_DIXON_COLES
        if not path.exists():
            raise FileNotFoundError(
                f"Dixon-Coles model not found: {path}\n"
                "Run: python src/dixon_coles.py"
            )
        return joblib.load(path)

    def __repr__(self):
        if self.fitted_:
            return (f"DixonColesModel(teams={len(self.teams_)}, "
                    f"home_adv={self.home_advantage_:.3f}, "
                    f"rho={self.rho_:.4f}, xi={self.xi})")
        return "DixonColesModel(unfitted)"


# ═══════════════════════════════════════════════════════
# SECTION 5 — INTEGRATE DC PREDICTIONS AS FEATURES
# ═══════════════════════════════════════════════════════

def add_dixon_coles_features(
    master_df: pd.DataFrame,
    league_key: str | None = None,
    refitting_window_seasons: int = 3,
) -> pd.DataFrame:
    """
    Add Dixon-Coles predictions as features to the master DataFrame.

    CRITICAL: For each match, the model is fitted on data BEFORE that match.
    We use a rolling fit — refit at the start of each season.

    This is the most important function for avoiding leakage:
    we never use future match results to fit the model.

    Args:
        master_df:                Full master DataFrame
        league_key:               Filter to one league (None = all, fits per league)
        refitting_window_seasons: Number of past seasons to use for fitting

    Returns:
        DataFrame with dc_* columns added.
    """
    df       = master_df.copy()
    df["date"] = pd.to_datetime(df["date"])

    # Process each league separately
    leagues = ([league_key] if league_key
               else df["league_key"].unique().tolist())

    all_frames = []

    for lk in leagues:
        league_df = df[df["league_key"] == lk].sort_values("date").copy()
        seasons   = sorted(league_df["season"].unique())

        dc_cols = {
            "dc_lambda_home": np.nan, "dc_lambda_away": np.nan,
            "dc_p_home": np.nan, "dc_p_draw": np.nan, "dc_p_away": np.nan,
            "dc_p_btts": np.nan, "dc_p_over_25": np.nan,
            "dc_p_over_15": np.nan, "dc_exp_goals": np.nan,
        }
        for col, default in dc_cols.items():
            league_df[col] = default

        for i, season in enumerate(seasons):
            # Fit on past seasons only
            past_seasons = seasons[max(0, i - refitting_window_seasons):i]
            if not past_seasons:
                # First season — no history, skip
                continue

            train_df = league_df[league_df["season"].isin(past_seasons)]
            if len(train_df) < 50:
                continue

            dc = DixonColesModel(xi=0.005)
            try:
                dc.fit(train_df, league_key=lk)
            except Exception as e:
                logger.warning(f"DC fit failed for {lk} season {season}: {e}")
                continue

            # Apply predictions to THIS season's matches (no leakage)
            season_mask = league_df["season"] == season
            season_rows = league_df[season_mask]

            for idx, row in season_rows.iterrows():
                try:
                    pred = dc.predict_probs(
                        row["home_team"], row["away_team"],
                        include_scorelines=False
                    )
                    for col in dc_cols:
                        field = col.replace("dc_", "")
                        if field == "exp_goals":
                            field = "expected_total_goals"
                        league_df.loc[idx, col] = pred.get(field, np.nan)
                except Exception:
                    pass

            logger.info(
                f"DC features: {lk} season={season} | "
                f"fitted on {len(train_df):,} matches"
            )

        all_frames.append(league_df)

    result = pd.concat(all_frames, ignore_index=True)
    result.sort_values(["league_key", "date"], inplace=True)
    result.reset_index(drop=True, inplace=True)

    coverage = result["dc_p_home"].notna().mean()
    logger.info(f"DC feature coverage: {coverage:.1%}")
    return result


# ═══════════════════════════════════════════════════════
# MAIN — train and save the model
# ═══════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Train and save the Dixon-Coles model"
    )
    parser.add_argument("--league", default=None,
                        help="Train on one league only e.g. EPL")
    parser.add_argument("--seasons", nargs="+", default=None,
                        help="Training seasons e.g. 2122 2223 2324")
    parser.add_argument("--show-strengths", action="store_true",
                        help="Print team strength table after fitting")
    args = parser.parse_args()

    PATHS.create_all()

    # Load master data
    master_path = PATHS.PROCESSED / "master.parquet"
    if not master_path.exists():
        logger.error("master.parquet not found. Run: python src/update_data.py")
        sys.exit(1)

    # FIX: Pylance stubs missing 'pyarrow' — suppress false positive
    master = pd.read_parquet(master_path, engine="pyarrow")  # type: ignore[call-overload]
    master["date"] = pd.to_datetime(master["date"])

    # Train
    dc = DixonColesModel(xi=0.005)
    dc.fit(
        master,
        league_key=args.league,
        seasons=args.seasons or SEASONS.TRAIN + SEASONS.VALIDATION,
    )
    dc.save()

    if args.show_strengths:
        strengths = dc.team_strengths()
        print("\n── Team Strengths (top 15 by attack) ──")
        print(strengths.head(15).to_string(index=False))

    # Quick prediction demo
    print("\n── Sample Prediction: Arsenal vs Chelsea ──")
    try:
        pred = dc.predict_probs("Arsenal", "Chelsea")
        print(f"  λ_home={pred['lambda_home']:.2f}  λ_away={pred['lambda_away']:.2f}")
        print(f"  1X2:   H={pred['p_home']:.1%}  D={pred['p_draw']:.1%}  A={pred['p_away']:.1%}")
        print(f"  BTTS:  {pred['p_btts']:.1%}")
        print(f"  O2.5:  {pred['p_over_25']:.1%}")
        print(f"  Exp goals: {pred['expected_total_goals']:.2f}")
        print("\n  Top scorelines:")
        for (h, a), p in list(pred["scorelines"].items())[:8]:
            print(f"    {h}-{a}:  {p:.1%}")
    except Exception as e:
        logger.warning(f"Demo prediction failed: {e}")