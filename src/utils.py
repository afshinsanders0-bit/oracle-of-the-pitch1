"""
src/utils.py — Shared Utilities & Common Functions
====================================================
Consolidates repeated code across modules:
- Column name sanitization
- API error handling with retries
- Data validation
- Logging setup

This module reduces code duplication and makes the codebase maintainable.
"""

import re
import sys
import time
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple, Union, Callable
from functools import wraps

import pandas as pd
import numpy as np
import requests
from loguru import logger


# ═══════════════════════════════════════════════════════
# COLUMN SANITIZATION
# ═══════════════════════════════════════════════════════

def sanitise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove characters that XGBoost rejects from column names.
    
    XGBoost doesn't accept special chars, spaces, or brackets in column names.
    This function standardizes all columns to alphanumeric + underscore.
    Handles name collisions by appending _1, _2, etc.
    
    Args:
        df: DataFrame with column names to sanitize
        
    Returns:
        DataFrame with cleaned column names
        
    Example:
        >>> df = pd.DataFrame({"Home Goals [FT]": [1, 2], "Away/Goals": [0, 1]})
        >>> clean = sanitise_columns(df)
        >>> clean.columns.tolist()
        ['Home_Goals__FT', 'Away_Goals']
    """
    seen: Dict[str, int] = {}
    new_cols: Dict[str, str] = {}
    
    for col in df.columns:
        # Replace invalid chars with underscore
        clean = re.sub(r'[^a-zA-Z0-9_]', '_', str(col))
        
        # Handle duplicates
        if clean in seen:
            seen[clean] += 1
            clean = f"{clean}_{seen[clean]}"
        else:
            seen[clean] = 0
            
        new_cols[col] = clean
    
    return df.rename(columns=new_cols)


def is_valid_column_name(col: str) -> bool:
    """
    Check if a column name is valid for XGBoost.
    
    Args:
        col: Column name to validate
        
    Returns:
        True if valid, False otherwise
    """
    return bool(re.match(r'^[a-zA-Z0-9_]+$', str(col)))


# ═══════════════════════════════════════════════════════
# API CALLS WITH RETRIES & ERROR HANDLING
# ═══════════════════════════════════════════════════════

def retry_on_exception(
    max_retries: int = 3,
    delay: float = 2.0,
    backoff_factor: float = 2.0,
    exceptions: Tuple = (requests.RequestException, ConnectionError)
) -> Callable:
    """
    Decorator for API calls with exponential backoff retry logic.
    
    Args:
        max_retries: Maximum number of retry attempts
        delay: Initial delay between retries (seconds)
        backoff_factor: Multiply delay by this after each retry
        exceptions: Tuple of exceptions to catch and retry on
        
    Returns:
        Decorated function that retries on exception
        
    Example:
        >>> @retry_on_exception(max_retries=3)
        ... def fetch_data(url):
        ...     return requests.get(url)
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            current_delay = delay
            last_exception = None
            
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        logger.warning(
                            f"{func.__name__} failed (attempt {attempt + 1}/{max_retries}). "
                            f"Retrying in {current_delay:.1f}s: {str(e)}"
                        )
                        time.sleep(current_delay)
                        current_delay *= backoff_factor
                    else:
                        logger.error(
                            f"{func.__name__} failed after {max_retries} attempts. "
                            f"Last error: {str(e)}"
                        )
            
            raise last_exception
        
        return wrapper
    return decorator


@retry_on_exception(max_retries=3, delay=2.0, backoff_factor=2.0)
def fetch_with_timeout(
    url: str,
    headers: Optional[Dict[str, str]] = None,
    timeout: int = 15
) -> requests.Response:
    """
    Fetch URL with timeout and retry logic.
    
    Args:
        url: URL to fetch
        headers: HTTP headers (optional)
        timeout: Request timeout in seconds
        
    Returns:
        Response object
        
    Raises:
        requests.RequestException: If all retries fail
    """
    response = requests.get(url, headers=headers, timeout=timeout)
    response.raise_for_status()
    return response


# ═══════════════════════════════════════════════════════
# DATA VALIDATION
# ═══════════════════════════════════════════════════════

def validate_dataframe(
    df: pd.DataFrame,
    required_columns: List[str],
    min_rows: int = 1
) -> bool:
    """
    Validate that a DataFrame has required structure.
    
    Args:
        df: DataFrame to validate
        required_columns: List of required column names
        min_rows: Minimum number of rows required
        
    Returns:
        True if valid, raises ValueError if not
        
    Raises:
        ValueError: If validation fails
    """
    if df is None or df.empty:
        raise ValueError("DataFrame is empty or None")
    
    if len(df) < min_rows:
        raise ValueError(
            f"DataFrame has {len(df)} rows, but {min_rows} required"
        )
    
    missing_cols = set(required_columns) - set(df.columns)
    if missing_cols:
        raise ValueError(
            f"Missing required columns: {missing_cols}\n"
            f"Available: {list(df.columns)}"
        )
    
    return True


def check_feature_mismatch(
    df: pd.DataFrame,
    expected_features: List[str]
) -> List[str]:
    """
    Check if DataFrame has all expected features and return missing ones.
    
    Args:
        df: DataFrame to check
        expected_features: List of expected feature names
        
    Returns:
        List of missing feature names (empty if all present)
    """
    missing = [f for f in expected_features if f not in df.columns]
    
    if missing:
        logger.warning(
            f"Missing {len(missing)}/{len(expected_features)} features: {missing}"
        )
    
    return missing


# ═══════════════════════════════════════════════════════
# PANDAS UTILITIES
# ═══════════════════════════════════════════════════════

def safe_fillna(df: pd.DataFrame, value: Any = -999.0) -> pd.DataFrame:
    """
    Safely fill NaN values with a sentinel value.
    
    XGBoost handles -999.0 as missing data better than NaN in some cases.
    This is safer than `df.fillna(value)` because it won't overflow integer columns.
    
    Args:
        df: DataFrame to fill
        value: Value to use for NaN (-999.0 for ML models)
        
    Returns:
        DataFrame with NaNs replaced
    """
    df_copy = df.copy()
    
    # For numeric columns, use the provided value
    numeric_cols = df_copy.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df_copy[col] = df_copy[col].fillna(value)
    
    # For non-numeric, use 'MISSING'
    non_numeric_cols = df_copy.select_dtypes(exclude=[np.number]).columns
    for col in non_numeric_cols:
        df_copy[col] = df_copy[col].fillna('MISSING')
    
    return df_copy


def align_features(
    df: pd.DataFrame,
    expected_features: List[str],
    fill_value: float = -999.0
) -> pd.DataFrame:
    """
    Align DataFrame to expected features for ML model input.
    
    - Adds missing features (filled with fill_value)
    - Reorders to match expected_features
    - Removes extra features
    
    Args:
        df: DataFrame with features
        expected_features: List of feature names model expects
        fill_value: Value to use for missing features
        
    Returns:
        DataFrame with aligned features
    """
    df_aligned = df.copy()
    
    # Add missing features
    for feat in expected_features:
        if feat not in df_aligned.columns:
            df_aligned[feat] = fill_value
    
    # Reorder and subset to expected features
    df_aligned = df_aligned[expected_features]
    
    # Fill any remaining NaNs
    df_aligned = safe_fillna(df_aligned, value=fill_value)
    
    return df_aligned


# ═══════════════════════════════════════════════════════
# LOGGING SETUP
# ═══════════════════════════════════════════════════════

def setup_logging(
    name: str,
    level: str = "INFO",
    log_file: Optional[Path] = None
) -> object:
    """
    Configure logger for a module with consistent format.
    
    Args:
        name: Module/logger name
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Path to log file (optional)
        
    Returns:
        Configured logger instance
        
    Example:
        >>> log = setup_logging(__name__, level="DEBUG")
        >>> log.debug("Debug message")
    """
    log_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{line}</cyan> — "
        "<level>{message}</level>"
    )
    
    # Remove default handler
    logger.remove()
    
    # Add stderr handler
    logger.add(
        sys.stderr,
        level=level,
        format=log_format,
        colorize=True,
    )
    
    # Add file handler if specified
    if log_file:
        logger.add(
            log_file,
            level=level,
            format=log_format,
            rotation="10 MB",
            retention="30 days",
        )
    
    return logger


# ═══════════════════════════════════════════════════════
# MISCELLANEOUS
# ═══════════════════════════════════════════════════════

def safe_divide(
    numerator: np.ndarray,
    denominator: np.ndarray,
    fill: float = 0.0
) -> np.ndarray:
    """
    Safely divide arrays, avoiding division by zero.
    
    Args:
        numerator: Numerator array
        denominator: Denominator array
        fill: Value to use where denominator is zero
        
    Returns:
        Result array
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.divide(numerator, denominator)
        result[~np.isfinite(result)] = fill
    return result


def format_percent(value: float, decimals: int = 1) -> str:
    """
    Format a decimal value as percentage string.
    
    Args:
        value: Decimal value (0.5 = 50%)
        decimals: Number of decimal places
        
    Returns:
        Formatted percentage string (e.g., "50.0%")
    """
    return f"{value * 100:.{decimals}f}%"


def format_odds(prob: float, decimals: int = 2) -> float:
    """
    Convert probability to decimal odds (European format).
    
    Formula: Odds = 1 / Probability
    
    Args:
        prob: Probability (0.0 to 1.0)
        decimals: Rounding precision
        
    Returns:
        Decimal odds
    """
    if prob <= 0 or prob >= 1:
        return np.nan
    return round(1.0 / prob, decimals)


# ═══════════════════════════════════════════════════════
# STARTUP CHECK
# ═══════════════════════════════════════════════════════

if __name__ == "__main__":
    print("✓ utils.py loaded successfully")
    
    # Test sanitise_columns
    test_df = pd.DataFrame({
        "Home Goals [FT]": [1, 2],
        "Away/Goals": [0, 1],
    })
    clean = sanitise_columns(test_df)
    print(f"✓ Column sanitization: {list(clean.columns)}")
    
    # Test format functions
    print(f"✓ 0.625 → {format_percent(0.625)} probability")
    print(f"✓ 0.50 → {format_odds(0.50)} odds")