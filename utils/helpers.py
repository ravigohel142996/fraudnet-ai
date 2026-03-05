"""
utils/helpers.py — Shared utility functions for FraudNet AI.

Keeps all generic, reusable helpers in one place so other modules stay clean.
"""

from __future__ import annotations

import hashlib
import logging
import time
from functools import wraps
from typing import Any, Callable

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def configure_logging(level: int = logging.INFO) -> None:
    """Configure root logger with a sensible format."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


# ---------------------------------------------------------------------------
# Timing decorator
# ---------------------------------------------------------------------------

def timed(func: Callable) -> Callable:
    """Decorator that logs the execution time of any function."""

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        logger.debug("%-40s completed in %.3f s", func.__qualname__, elapsed)
        return result

    return wrapper


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def generate_account_id(index: int) -> str:
    """Return a deterministic, human-readable account identifier."""
    digest = hashlib.sha1(str(index).encode()).hexdigest()[:6].upper()  # noqa: S324
    return f"ACC-{digest}"


def encode_cyclical(series: pd.Series, period: float) -> tuple[pd.Series, pd.Series]:
    """Sine/cosine encoding for a cyclical numeric feature (e.g. hour-of-day).

    Args:
        series: Raw numeric values (e.g. 0-23 for hours).
        period: Full cycle length (e.g. 24 for hours).

    Returns:
        Tuple of (sin_encoded, cos_encoded) Series.
    """
    angle = 2.0 * np.pi * series / period
    return np.sin(angle), np.cos(angle)


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Return numerator / denominator, or *default* when denominator is zero."""
    return numerator / denominator if denominator else default


def clip_outliers(series: pd.Series, lower_q: float = 0.01, upper_q: float = 0.99) -> pd.Series:
    """Winsorise a numeric series to [lower_q, upper_q] quantiles."""
    lo = series.quantile(lower_q)
    hi = series.quantile(upper_q)
    return series.clip(lower=lo, upper=hi)


# ---------------------------------------------------------------------------
# DataFrame helpers
# ---------------------------------------------------------------------------

def summarise_dataframe(df: pd.DataFrame) -> dict[str, Any]:
    """Return a dictionary of high-level statistics about a DataFrame."""
    return {
        "rows": len(df),
        "columns": len(df.columns),
        "missing_pct": df.isnull().mean().mean() * 100,
        "memory_mb": df.memory_usage(deep=True).sum() / 1024**2,
    }


def stratified_sample(
    df: pd.DataFrame,
    label_col: str,
    n: int,
    random_state: int = 42,
) -> pd.DataFrame:
    """Return a stratified random sample of size *n* from *df*."""
    return (
        df.groupby(label_col, group_keys=False)
        .apply(lambda g: g.sample(min(len(g), n // df[label_col].nunique()), random_state=random_state))
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def fmt_number(value: float, decimals: int = 2) -> str:
    """Format a float for display (e.g. 1_234_567.89 → '1,234,567.89')."""
    return f"{value:,.{decimals}f}"


def fmt_percent(value: float, decimals: int = 1) -> str:
    """Format a ratio as a percentage string (e.g. 0.0532 → '5.3%')."""
    return f"{value * 100:.{decimals}f}%"
