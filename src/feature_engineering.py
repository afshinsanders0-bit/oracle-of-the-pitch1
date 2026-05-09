import pandas as pd
import numpy as np

def build_features(df):
    """
    Transforms raw match data into model-ready features.
    Updated to match the expected name in predict.py.
    """
    # 1. Sort values to ensure rolling calculations are chronological
    df = df.sort_values(['team', 'date'])

    # 2. Example: Create Rolling Averages for Goals
    # This helps the model see recent form rather than just season totals
    df['rolling_gs'] = df.groupby('team')['goals_scored'].transform(
        lambda x: x.rolling(window=5, min_periods=1).mean()
    )
    df['rolling_ga'] = df.groupby('team')['goals_against'].transform(
        lambda x: x.rolling(window=5, min_periods=1).mean()
    )

    # 3. Example: Win Streak calculation
    df['result_value'] = df['result'].map({'W': 1, 'D': 0, 'L': -1})
    df['form_index'] = df.groupby('team')['result_value'].transform(
        lambda x: x.rolling(window=3, min_periods=1).sum()
    )

    # 4. Handle missing values created by rolling windows
    df = df.fillna(0)

    return df

def get_feature_list():
    """
    Returns a list of column names that should be used as model inputs.
    """
    return ['rolling_gs', 'rolling_ga', 'form_index']

# --- COMPATIBILITY LAYER ---
# This ensures that if any other file still calls 'load_features', 
# it will point to our new 'build_features' function without crashing.
load_features = build_features