"""
src/live_prediction_handler.py — Live Fixture Prediction Handler
=================================================================
Connects trained ML models to live fixtures. Drop-in ready.

Usage:
    from src.live_prediction_handler import LivePredictionHandler
    
    handler = LivePredictionHandler()
    handler.load_model()
    handler.load_historical_data(df)
    predictions = handler.predict_batch(fixtures)
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
import joblib
import logging

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import PATHS
from src.utils import align_features, sanitise_columns

logger = logging.getLogger(__name__)


class LivePredictionHandler:
    """Load trained models and predict on live fixtures."""
    
    def __init__(self) -> None:
        self.model: Optional[Any] = None
        self.feature_names: Optional[List[str]] = None
        self.historical_df: Optional[pd.DataFrame] = None
    
    def load_model(self, model_path: Optional[Path] = None) -> bool:
        """Load XGBoost model and feature list."""
        try:
            mpath = model_path or PATHS.MODEL_MATCH_RESULT
            if not mpath.exists():
                logger.error(f"✗ Model not found: {mpath}")
                return False
            
            self.model = joblib.load(mpath)
            
            for fname in ["match_result_current_features.pkl", "match_result_features.pkl"]:
                fpath = PATHS.MODELS / fname
                if fpath.exists():
                    self.feature_names = joblib.load(fpath)
                    feature_count = len(self.feature_names) if self.feature_names is not None else 0
                    logger.info(f"✓ Model loaded with {feature_count} features")
                    return True
            
            logger.error("✗ No feature list found")
            return False
        except Exception as e:
            logger.error(f"✗ Failed to load model: {e}")
            return False
    
    def load_historical_data(self, df: Optional[pd.DataFrame] = None) -> bool:
        """Load historical data for feature engineering."""
        try:
            if df is not None:
                self.historical_df = df.copy()
                logger.info(f"✓ Loaded {len(df):,} historical matches")
                return True
            
            for fname in ["features_master.parquet", "master.parquet"]:
                fpath = PATHS.PROCESSED / fname
                if fpath.exists():
                    # Use cast to suppress type checker error for valid pandas engine param
                    self.historical_df = pd.read_parquet(
                        fpath, 
                        engine="pyarrow"  # type: ignore[arg-type]
                    )
                    logger.info(f"✓ Loaded {len(self.historical_df):,} historical matches")
                    return True
            
            logger.warning("⚠ No historical data (features will use -999 for missing)")
            return True
        except Exception as e:
            logger.error(f"✗ Failed to load historical data: {e}")
            return False
    
    def _build_features(self, fixture: Dict[str, Any]) -> Dict[str, float]:
        """Build feature dict for a fixture."""
        feat: Dict[str, float] = {}
        
        # Current stats
        feat["home_shots"] = float(fixture.get("home_shots", -999.0))
        feat["away_shots"] = float(fixture.get("away_shots", -999.0))
        feat["home_corners"] = float(fixture.get("home_corners", -999.0))
        feat["away_corners"] = float(fixture.get("away_corners", -999.0))
        
        # Derived
        feat["shot_diff"] = feat["home_shots"] - feat["away_shots"] if feat["home_shots"] > -999 else -999.0
        feat["corner_diff"] = feat["home_corners"] - feat["away_corners"] if feat["home_corners"] > -999 else -999.0
        
        # H2H (if historical data available)
        if self.historical_df is not None:
            h2h = self._get_h2h(fixture["home_team"], fixture["away_team"])
            feat.update(h2h)
        
        # Fill missing features with -999
        if self.feature_names is not None:
            for fname in self.feature_names:
                if fname not in feat:
                    feat[fname] = -999.0
        
        return feat
    
    def _get_h2h(self, home_team: str, away_team: str) -> Dict[str, float]:
        """Get head-to-head stats."""
        try:
            if self.historical_df is None:
                return {"h2h_matches": 0.0, "h2h_home_wins": 0.0}
            
            h2h_matches = self.historical_df[
                ((self.historical_df["home_team"] == home_team) & (self.historical_df["away_team"] == away_team)) |
                ((self.historical_df["home_team"] == away_team) & (self.historical_df["away_team"] == home_team))
            ].tail(20)
            
            if h2h_matches.empty:
                return {"h2h_matches": 0.0, "h2h_home_wins": 0.0}
            
            home_view = h2h_matches[h2h_matches["home_team"] == home_team]
            wins = (home_view["result"] == "H").sum() if not home_view.empty else 0
            
            return {
                "h2h_matches": float(len(h2h_matches)),
                "h2h_home_wins": float(wins),
            }
        except Exception:
            return {"h2h_matches": 0.0, "h2h_home_wins": 0.0}
    
    def predict(self, fixture: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Predict on single fixture."""
        if self.model is None or self.feature_names is None:
            return None
        
        try:
            feat_dict = self._build_features(fixture)
            feat_df = pd.DataFrame([feat_dict])
            feat_df = sanitise_columns(feat_df)
            feat_df = align_features(feat_df, self.feature_names, fill_value=-999.0)
            feat_df = feat_df.astype(np.float32)
            
            probas_result = self.model.predict_proba(feat_df)
            
            # Ensure probas is not None and has the expected shape
            if probas_result is None or len(probas_result) == 0:
                logger.warning("Model returned empty probabilities")
                return None
            
            probas: np.ndarray = probas_result[0]
            max_prob = probas.max()
            
            confidence = "HIGH" if max_prob >= 0.65 else "MEDIUM" if max_prob >= 0.50 else "LOW"
            top_idx = int(np.argmax(probas))
            top_picks = ["Home Win", "Draw", "Away Win"]
            
            return {
                "home_win_prob": float(probas[0]),
                "draw_prob": float(probas[1]),
                "away_win_prob": float(probas[2]),
                "confidence": confidence,
                "top_pick": top_picks[top_idx],
                "top_probability": float(max_prob),
            }
        except Exception as e:
            logger.warning(f"Prediction failed: {e}")
            return None
    
    def predict_batch(self, fixtures: List[Dict[str, Any]]) -> Dict[str, Optional[Dict[str, Any]]]:
        """Predict on multiple fixtures."""
        results: Dict[str, Optional[Dict[str, Any]]] = {}
        for fix in fixtures:
            fix_id = fix.get("id", f"{fix.get('home_team')}_{fix.get('away_team')}")
            results[fix_id] = self.predict(fix)
        return results
    
    def is_ready(self) -> bool:
        """Check if handler can make predictions."""
        return self.model is not None and self.feature_names is not None