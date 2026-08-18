"""
update_data.py — Full Data Pipeline Entry Point
================================================
Orchestrates the complete offline pipeline:

    1. Download raw CSVs from football-data.co.uk
    2. Parse, clean, merge into master.parquet
    3. Add ELO ratings (per-league, chronological)
    4. Scrape xG from Understat (optional)
    5. Build rolling features
    6. Train all market models

Run:
    python src/update_data.py
    python src/update_data.py --leagues EPL LA_LIGA --seasons 2324 2425
    python src/update_data.py --skip-xg          # skip Understat scraping
    python src/update_data.py --train-only       # skip download, just train
"""

import sys
import argparse
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import pandas as pd
import numpy as np
from loguru import logger

from config import PATHS, SEASONS, LOGGING, RANDOM_SEED
from src.data_loader import build_master_df, save_master_df, load_master_df
from src.elo import add_elo_ratings, save_elo_df
from src.xg_scraper import fetch_all_xg, save_xg_data, add_xg_features
from src.feature_engineering import build_features
from src.train import train_model
from src.trainCurrentOnly import main as train_current_only
from src.dixon_coles import DixonColesModel

np.random.seed(RANDOM_SEED)

logger.remove()
logger.add(
    sys.stderr,
    level=LOGGING.LEVEL,
    format=LOGGING.FORMAT,
    colorize=True,
)
logger.add(
    PATHS.LOG_FILE,
    level="DEBUG",
    format=LOGGING.FORMAT,
    rotation=LOGGING.ROTATION,
    retention=LOGGING.RETENTION,
)


def run_pipeline(
    leagues: list[str] | None = None,
    seasons: list[str] | None = None,
    skip_xg: bool = False,
    train_only: bool = False,
) -> None:
    PATHS.create_all()

    if not train_only:
        logger.info("=" * 60)
        logger.info("STEP 1 — Download & build master DataFrame")
        logger.info("=" * 60)
        master_df = build_master_df(leagues=leagues, seasons=seasons)
        save_master_df(master_df)
    else:
        logger.info("Skipping download — loading existing master.parquet")
        master_df = load_master_df()
        if seasons:
            master_df = master_df[master_df["season"].isin(seasons)]

    logger.info("")
    logger.info("=" * 60)
    logger.info("STEP 2 — Add ELO ratings")
    logger.info("=" * 60)
    master_elo = add_elo_ratings(master_df, per_league=True)
    save_elo_df(master_elo)

    if not skip_xg:
        logger.info("")
        logger.info("=" * 60)
        logger.info("STEP 3 — Fetch xG data from Understat")
        logger.info("=" * 60)
        try:
            xg_df = fetch_all_xg(leagues=leagues, seasons=seasons)
            save_xg_data(xg_df)
            master_elo = add_xg_features(master_elo, xg_df=xg_df)
        except Exception as e:
            logger.warning(f"xG step failed: {e} — continuing without xG")
    else:
        logger.info("Skipping xG scraping (--skip-xg)")

    logger.info("")
    logger.info("=" * 60)
    logger.info("STEP 4 — Build rolling features")
    logger.info("=" * 60)
    features_df = build_features(master_elo)
    features_path = PATHS.PROCESSED / "features_master.parquet"
    features_df.to_parquet(features_path, index=False, engine="pyarrow")  # type: ignore[call-overload]
    logger.success(f"Saved: {features_path}  ({len(features_df):,} rows)")

    logger.info("")
    logger.info("=" * 60)
    logger.info("STEP 5 — Train match result model (XGBoost)")
    logger.info("=" * 60)
    train_model(seasons=seasons, save=True)

    logger.info("")
    logger.info("=" * 60)
    logger.info("STEP 6 — Train current-only models (BTTS, O/U, Corners)")
    logger.info("=" * 60)
    train_current_only()

    logger.info("")
    logger.info("=" * 60)
    logger.info("STEP 7 — Train Dixon-Coles model")
    logger.info("=" * 60)
    dc = DixonColesModel(xi=0.005)
    dc.fit(
        features_df,
        seasons=(seasons or SEASONS.TRAIN + SEASONS.VALIDATION),
    )
    dc.save()

    logger.success("=" * 60)
    logger.success("PIPELINE COMPLETE")
    logger.success("=" * 60)
    logger.success(f"Features : {features_path}")
    logger.success(f"Model    : {PATHS.MODEL_MATCH_RESULT}")
    logger.success(f"DC model : {PATHS.MODEL_DIXON_COLES}")
    logger.success("Run: streamlit run app.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run full data + training pipeline")
    parser.add_argument("--leagues", nargs="+", default=None, help="Leagues to process")
    parser.add_argument("--seasons", nargs="+", default=None, help="Seasons to process")
    parser.add_argument("--skip-xg", action="store_true", help="Skip Understat xG scraping")
    parser.add_argument("--train-only", action="store_true", help="Skip download, just train")
    args = parser.parse_args()

    run_pipeline(
        leagues=args.leagues,
        seasons=args.seasons,
        skip_xg=args.skip_xg,
        train_only=args.train_only,
    )
