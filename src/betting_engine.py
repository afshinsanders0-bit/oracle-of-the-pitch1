"""
betting_engine.py — Sports Investing Engine + Bet Builder
==========================================================
PURPOSE:
    Full bet builder that:
    - Calculates Expected Value (EV) per selection across all markets
    - Applies Fractional Kelly Criterion for stake sizing
    - Detects value bets (edge >= MIN_EDGE over fair odds)
    - Builds 2-leg and 3-leg accumulators from value selections
    - Suggests the BEST single bet and best accumulator per match
    - Returns structured output ready for the Streamlit dashboard

USAGE:
    from src.betting_engine import BetBuilder
    builder = BetBuilder(bankroll=1000.0)
    result  = builder.build(match_predictions)

    # match_predictions is a dict keyed by market:
    # {
    #   "match_result": {"Home Win": 0.52, "Draw": 0.27, "Away Win": 0.21},
    #   "btts":         {"Yes": 0.61, "No": 0.39},
    #   "over_under":   {"Over 2.5": 0.58, "Under 2.5": 0.42},
    #   "corners":      {"Over 9.5": 0.55, "Under 9.5": 0.45},
    # }
    # market_odds is a parallel dict of bookmaker odds (decimal):
    # {
    #   "match_result": {"Home Win": 2.10, "Draw": 3.40, "Away Win": 3.20},
    #   "btts":         {"Yes": 1.72, "No": 2.05},
    #   ...
    # }
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from itertools import combinations
from typing import Optional
from config import VALUE_BET, KELLY


# ═══════════════════════════════════════════════════════
# SECTION 1 — CORE MATH
# ═══════════════════════════════════════════════════════

def calculate_ev(model_prob: float, decimal_odds: float) -> float:
    """
    Expected Value = (model_prob × decimal_odds) - 1
    Positive EV = long-run profit potential.
    e.g. EV=+0.08 means we expect 8% return per unit staked.
    """
    if decimal_odds <= 1.0 or model_prob <= 0:
        return -1.0
    return round((model_prob * decimal_odds) - 1.0, 4)


def kelly_fraction(model_prob: float, decimal_odds: float) -> float:
    """
    Fractional Kelly: 25% of full Kelly, capped at MAX_STAKE_PCT.
    Full Kelly: f = (b*p - q) / b
    Returns 0.0 if no edge exists.
    """
    if decimal_odds <= 1.0 or model_prob <= 0:
        return 0.0
    b          = decimal_odds - 1.0
    p          = model_prob
    q          = 1.0 - p
    full_kelly = max(0.0, (b * p - q) / b)
    frac       = full_kelly * KELLY.FRACTION
    return round(min(frac, KELLY.MAX_STAKE_PCT), 6)


def kelly_stake_units(model_prob: float, decimal_odds: float, bankroll: float) -> float:
    """Return recommended stake in bankroll units (not percentage)."""
    return round(kelly_fraction(model_prob, decimal_odds) * bankroll, 2)


def remove_margin(odds_dict: dict[str, float]) -> dict[str, float]:
    """
    Remove bookmaker overround from a set of odds to get fair probabilities.
    overround = sum(1/odds) — typically 1.05–1.10.
    Returns dict of {outcome: fair_probability}.
    """
    implied = {k: 1 / v for k, v in odds_dict.items() if v > 1.0}
    total   = sum(implied.values())
    if total == 0:
        return {k: 0.0 for k in odds_dict}
    return {k: round(v / total, 4) for k, v in implied.items()}


def confidence_tier(prob: float) -> str:
    """HIGH >65% | MEDIUM 50-65% | LOW <50%"""
    if prob >= VALUE_BET.HIGH_CONFIDENCE:
        return "HIGH"
    elif prob >= VALUE_BET.MEDIUM_CONFIDENCE:
        return "MEDIUM"
    return "LOW"


# ═══════════════════════════════════════════════════════
# SECTION 2 — DATA CLASSES
# ═══════════════════════════════════════════════════════

@dataclass
class BetSelection:
    """A single evaluated market selection (one leg)."""
    market:       str     # e.g. 'Match Result'
    outcome:      str     # e.g. 'Home Win'
    model_prob:   float   # Our model's probability
    fair_prob:    float   # Fair probability (overround removed)
    odds:         float   # Decimal odds
    ev:           float   # Expected value
    edge:         float   # model_prob - fair_prob
    kelly_frac:   float   # Fractional Kelly fraction
    stake_units:  float   # Recommended stake in units
    confidence:   str     # HIGH / MEDIUM / LOW
    is_value:     bool    # True if edge >= MIN_EDGE

    def as_dict(self) -> dict:
        return {
            "market":       self.market,
            "outcome":      self.outcome,
            "model_prob":   round(self.model_prob, 4),
            "fair_prob":    round(self.fair_prob, 4),
            "odds":         self.odds,
            "ev":           round(self.ev, 4),
            "edge":         round(self.edge, 4),
            "kelly_frac":   round(self.kelly_frac, 4),
            "stake_units":  self.stake_units,
            "confidence":   self.confidence,
            "is_value":     self.is_value,
        }


@dataclass
class AccumulatorLeg:
    """One leg inside an accumulator."""
    market:     str
    outcome:    str
    odds:       float
    model_prob: float
    ev:         float


@dataclass
class Accumulator:
    """Multi-leg accumulator built from value selections."""
    legs:          list[AccumulatorLeg]
    combined_odds: float   # Product of all leg odds
    combined_prob: float   # Product of all model probabilities
    combined_ev:   float   # (combined_prob × combined_odds) - 1
    kelly_frac:    float
    stake_units:   float
    n_legs:        int

    def as_dict(self) -> dict:
        return {
            "n_legs":        self.n_legs,
            "combined_odds": round(self.combined_odds, 2),
            "combined_prob": round(self.combined_prob, 4),
            "combined_ev":   round(self.combined_ev, 4),
            "kelly_frac":    round(self.kelly_frac, 4),
            "stake_units":   self.stake_units,
            "legs": [
                {
                    "market":  l.market,
                    "outcome": l.outcome,
                    "odds":    l.odds,
                    "prob":    round(l.model_prob, 4),
                    "ev":      round(l.ev, 4),
                }
                for l in self.legs
            ],
        }


@dataclass
class BetBuilderResult:
    """Full output of the bet builder for one match."""
    home_team:       str
    away_team:       str
    league_key:      str
    all_selections:  list[BetSelection]       # All markets evaluated
    value_bets:      list[BetSelection]       # Only selections with positive edge
    best_single:     Optional[BetSelection]   # Single bet with highest EV
    best_double:     Optional[Accumulator]    # Best 2-leg acca
    best_treble:     Optional[Accumulator]    # Best 3-leg acca
    model_verdict:   str                      # e.g. "Home Win @ 62% — HIGH confidence"
    bankroll:        float

    def summary(self) -> dict:
        return {
            "match":         f"{self.home_team} vs {self.away_team}",
            "league":        self.league_key,
            "model_verdict": self.model_verdict,
            "value_bets":    len(self.value_bets),
            "best_single":   self.best_single.as_dict() if self.best_single else None,
            "best_double":   self.best_double.as_dict() if self.best_double else None,
            "best_treble":   self.best_treble.as_dict() if self.best_treble else None,
            "all_markets":   [s.as_dict() for s in self.all_selections],
        }


# ═══════════════════════════════════════════════════════
# SECTION 3 — BET BUILDER ENGINE
# ═══════════════════════════════════════════════════════

class BetBuilder:
    """
    Main bet builder.

    Given multi-market model predictions and bookmaker odds for one match,
    evaluates every selection and returns the best bets with Kelly stakes.

    Usage:
        builder = BetBuilder(bankroll=1000.0)
        result  = builder.build(
            home_team="Arsenal", away_team="Chelsea", league_key="EPL",
            predictions={
                "match_result": {"Home Win": 0.52, "Draw": 0.27, "Away Win": 0.21},
                "btts":         {"Yes": 0.61, "No": 0.39},
                "over_under":   {"Over 2.5": 0.58, "Under 2.5": 0.42},
            },
            market_odds={
                "match_result": {"Home Win": 2.10, "Draw": 3.40, "Away Win": 3.20},
                "btts":         {"Yes": 1.72, "No": 2.05},
                "over_under":   {"Over 2.5": 1.85, "Under 2.5": 1.95},
            }
        )
    """

    # Default odds when bookmaker odds are unavailable
    # (used to still show model probabilities without value flagging)
    DEFAULT_ODDS = {
        "match_result": {"Home Win": 2.5,  "Draw": 3.3,  "Away Win": 2.9},
        "btts":         {"Yes": 1.85,      "No": 1.90},
        "over_under":   {"Over 2.5": 1.85, "Under 2.5": 1.95},
        "corners":      {"Over 9.5": 1.83, "Under 9.5": 1.97},
    }

    MARKET_LABELS = {
        "match_result": "Match Result",
        "btts":         "Both Teams to Score",
        "over_under":   "Over/Under 2.5 Goals",
        "corners":      "Corners Over/Under 9.5",
    }

    def __init__(self, bankroll: float = 1000.0):
        self.bankroll = bankroll

    def build(
        self,
        home_team:    str,
        away_team:    str,
        league_key:   str,
        predictions:  dict[str, dict[str, float]],
        market_odds:  dict[str, dict[str, float]] | None = None,
    ) -> BetBuilderResult:
        """
        Run the full bet builder pipeline.

        Args:
            home_team:   Home team name
            away_team:   Away team name
            league_key:  League key e.g. 'EPL'
            predictions: Model probabilities per market per outcome
            market_odds: Bookmaker decimal odds per market per outcome
                         (uses defaults if None)

        Returns:
            BetBuilderResult with all selections and recommendations
        """
        if market_odds is None:
            market_odds = {}

        all_selections = []

        # ── Evaluate every market ──────────────────────────────────────────
        for market_key, probs in predictions.items():
            label  = self.MARKET_LABELS.get(market_key, market_key)
            odds_d = market_odds.get(market_key, self.DEFAULT_ODDS.get(market_key, {}))

            # Get fair probabilities (overround removed)
            fair_probs = remove_margin(odds_d) if odds_d else {}

            for outcome, model_prob in probs.items():
                odd       = odds_d.get(outcome, 0.0)
                fair_prob = fair_probs.get(outcome, 0.0)

                if odd <= 1.0 or model_prob <= 0:
                    continue

                edge      = round(model_prob - fair_prob, 4) if fair_prob > 0 else 0.0
                ev        = calculate_ev(model_prob, odd)
                kf        = kelly_fraction(model_prob, odd)
                stake     = kelly_stake_units(model_prob, odd, self.bankroll)
                conf      = confidence_tier(model_prob)
                is_value  = (
                    edge >= VALUE_BET.MIN_EDGE
                    and model_prob >= VALUE_BET.MIN_MODEL_PROB
                    and VALUE_BET.MIN_ODDS <= odd <= VALUE_BET.MAX_ODDS
                )

                all_selections.append(BetSelection(
                    market=label, outcome=outcome,
                    model_prob=model_prob, fair_prob=fair_prob,
                    odds=odd, ev=ev, edge=edge,
                    kelly_frac=kf, stake_units=stake,
                    confidence=conf, is_value=is_value,
                ))

        # ── Filter value bets ──────────────────────────────────────────────
        value_bets = sorted(
            [s for s in all_selections if s.is_value],
            key=lambda x: -x.ev
        )

        # ── Best single bet ────────────────────────────────────────────────
        best_single = value_bets[0] if value_bets else (
            # If no value bets, return highest EV selection anyway (marked non-value)
            max(all_selections, key=lambda x: x.ev) if all_selections else None
        )

        # ── Build accumulators from value selections ───────────────────────
        best_double = self._best_accumulator(value_bets, n_legs=2)
        best_treble = self._best_accumulator(value_bets, n_legs=3)

        # ── Model verdict (top prediction across all markets) ──────────────
        mr = predictions.get("match_result", {})
        if mr:
            # FIX: use explicit lambda instead of dict.get (avoids Pylance reportCallIssue)
            top_out  = max(mr, key=lambda k: mr[k])
            top_prob = mr[top_out]
            conf_str = confidence_tier(top_prob)
            model_verdict = f"{top_out} @ {top_prob:.0%} — {conf_str} confidence"
        else:
            model_verdict = "Insufficient data"

        return BetBuilderResult(
            home_team=home_team,
            away_team=away_team,
            league_key=league_key,
            all_selections=all_selections,
            value_bets=value_bets,
            best_single=best_single,
            best_double=best_double,
            best_treble=best_treble,
            model_verdict=model_verdict,
            bankroll=self.bankroll,
        )

    def _best_accumulator(
        self,
        selections: list[BetSelection],
        n_legs: int = 2,
    ) -> Optional[Accumulator]:
        """
        Build the best N-leg accumulator from value selections.

        Tries all combinations of N legs from the value selections list
        (max one selection per market to avoid correlated outcomes),
        picks the combination with highest combined EV.

        Args:
            selections: List of value BetSelection objects
            n_legs:     Number of legs in the accumulator

        Returns:
            Best Accumulator, or None if insufficient value selections
        """
        # Need at least N selections, and ideally from different markets
        if len(selections) < n_legs:
            return None

        best_acca  = None
        best_ev    = -999.0

        for combo in combinations(selections, n_legs):
            # Reject combos with duplicate markets (correlated bets)
            markets = [s.market for s in combo]
            if len(set(markets)) < n_legs:
                continue

            combined_odds = 1.0
            combined_prob = 1.0
            for s in combo:
                combined_odds *= s.odds
                combined_prob *= s.model_prob

            combined_ev = calculate_ev(combined_prob, combined_odds)
            kf          = kelly_fraction(combined_prob, combined_odds)
            stake       = kelly_stake_units(combined_prob, combined_odds, self.bankroll)

            if combined_ev > best_ev:
                best_ev   = combined_ev
                best_acca = Accumulator(
                    legs=[AccumulatorLeg(
                        market=s.market, outcome=s.outcome,
                        odds=s.odds, model_prob=s.model_prob, ev=s.ev
                    ) for s in combo],
                    combined_odds=round(combined_odds, 2),
                    combined_prob=round(combined_prob, 4),
                    combined_ev=round(combined_ev, 4),
                    kelly_frac=round(kf, 4),
                    stake_units=stake,
                    n_legs=n_legs,
                )

        return best_acca


# ═══════════════════════════════════════════════════════
# SECTION 4 — BATCH VALUE BET DETECTION
# ═══════════════════════════════════════════════════════

def detect_value_bets(predictions: pd.DataFrame) -> pd.DataFrame:
    """
    Detect value bets from a batch predictions DataFrame.

    predictions DataFrame must have columns:
        model_prob, odds, market, outcome, home_team, away_team

    Returns filtered DataFrame of value bets with EV and Kelly columns added.
    """
    df = predictions.copy()

    df["ev"] = df.apply(
        lambda x: calculate_ev(x["model_prob"], x["odds"]), axis=1
    )
    df["kelly_frac"] = df.apply(
        lambda x: kelly_fraction(x["model_prob"], x["odds"]), axis=1
    )
    df["recommended_stake_pct"] = df["kelly_frac"] * 100
    df["confidence"] = df["model_prob"].apply(confidence_tier)

    value_mask = (
        (df["ev"] > VALUE_BET.MIN_EDGE) &
        (df["model_prob"] > VALUE_BET.MIN_MODEL_PROB) &
        (df["odds"].between(VALUE_BET.MIN_ODDS, VALUE_BET.MAX_ODDS))
    )

    result = df[value_mask].sort_values("ev", ascending=False).copy()
    print(f"Found {len(result)} value bets with positive EV from {len(df)} selections")
    return result


# ═══════════════════════════════════════════════════════
# SECTION 5 — QUICK STAKE CALCULATOR
# ═══════════════════════════════════════════════════════

def quick_kelly(model_prob: float, decimal_odds: float, bankroll: float = 1000.0) -> dict:
    """
    Quick Kelly calculation for dashboard display.

    Returns dict with full breakdown.
    """
    b         = max(decimal_odds - 1.0, 0.0)
    p         = model_prob
    q         = 1.0 - p
    full_k    = max(0.0, (b * p - q) / b) if b > 0 else 0.0
    frac_k    = min(full_k * KELLY.FRACTION, KELLY.MAX_STAKE_PCT)
    stake     = round(frac_k * bankroll, 2)
    ev        = calculate_ev(p, decimal_odds)

    return {
        "full_kelly_pct":  round(full_k * 100, 2),
        "frac_kelly_pct":  round(frac_k * 100, 2),
        "stake_units":     stake,
        "stake_pct":       round(frac_k * 100, 2),
        "ev":              ev,
        "recommended":     stake > 0 and ev > 0,
        "verdict":         (
            f"Stake {stake:.1f} units ({frac_k*100:.1f}% of bankroll) | EV={ev:+.3f}"
            if stake > 0
            else "No edge — do not bet"
        ),
    }


print("✅ Betting Engine loaded")