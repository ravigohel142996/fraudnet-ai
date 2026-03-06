"""
utils.helpers
=============
Shared utility functions used across multiple modules.

Kept intentionally small — only genuinely reusable, stateless helpers live
here.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Generator

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def fmt_currency(value: float) -> str:
    """Format *value* as a USD currency string."""
    if value >= 1_000_000:
        return f"${value / 1_000_000:.2f}M"
    if value >= 1_000:
        return f"${value / 1_000:.1f}K"
    return f"${value:.2f}"


def fmt_percent(value: float, decimals: int = 2) -> str:
    """Format *value* (0–100) as a percentage string."""
    return f"{value:.{decimals}f}%"


def fmt_number(value: int | float) -> str:
    """Format *value* with thousands separators."""
    return f"{int(value):,}"


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def validate_dataframe(df: pd.DataFrame, required_columns: list[str]) -> None:
    """Raise ``ValueError`` if *df* is missing any *required_columns*."""
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise ValueError(f"DataFrame is missing required columns: {missing}")


# ---------------------------------------------------------------------------
# Timing helper
# ---------------------------------------------------------------------------


@contextmanager
def timer(label: str = "Operation") -> Generator[None, None, None]:
    """Context manager that prints elapsed time on exit."""
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        print(f"[timer] {label} completed in {elapsed:.3f}s")


# ---------------------------------------------------------------------------
# Data utilities
# ---------------------------------------------------------------------------


def safe_divide(numerator: float, denominator: float, fallback: float = 0.0) -> float:
    """Return *numerator / denominator*, falling back to *fallback* on ZeroDivisionError."""
    return numerator / denominator if denominator != 0 else fallback


def clamp(value: float, lo: float, hi: float) -> float:
    """Clamp *value* to the closed interval [*lo*, *hi*]."""
    return max(lo, min(hi, value))


def risk_label(score: float) -> str:
    """Convert a numeric risk score (0–1) to a human-readable label."""
    if score >= 0.75:
        return "Critical"
    if score >= 0.5:
        return "High"
    if score >= 0.25:
        return "Medium"
    return "Low"
