"""
analytics/fraud_metrics.py — Aggregated fraud analytics computations.

Computes KPIs, trends, and risk tables that power the dashboard.
All functions are pure (no side-effects) and return DataFrames or dicts.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from utils.helpers import fmt_percent, safe_divide, timed

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# KPI helpers
# ---------------------------------------------------------------------------

@timed
def compute_kpis(df: pd.DataFrame) -> dict:
    """Return top-level KPI values for the fraud overview section.

    Parameters
    ----------
    df:
        Enriched transaction DataFrame (must contain ``is_fraud`` and
        optionally ``fraud_probability`` columns).

    Returns
    -------
    dict
        Keys: ``total_transactions``, ``fraud_count``, ``fraud_rate``,
        ``total_amount``, ``fraud_amount``, ``high_risk_accounts``.
    """
    total = len(df)
    fraud_count = int(df["is_fraud"].sum())
    fraud_rate = safe_divide(fraud_count, total)

    total_amount = float(df["amount"].sum())
    fraud_amount = float(df.loc[df["is_fraud"] == 1, "amount"].sum())

    high_risk = 0
    if "fraud_probability" in df.columns:
        high_risk = int((df["fraud_probability"] >= 0.7).sum())

    return {
        "total_transactions": total,
        "fraud_count": fraud_count,
        "fraud_rate": fraud_rate,
        "fraud_rate_pct": fmt_percent(fraud_rate),
        "total_amount": total_amount,
        "fraud_amount": fraud_amount,
        "high_risk_accounts": high_risk,
    }


# ---------------------------------------------------------------------------
# Trend analysis
# ---------------------------------------------------------------------------

@timed
def compute_fraud_trend(df: pd.DataFrame, freq: str = "D") -> pd.DataFrame:
    """Aggregate fraud counts and rates over time.

    Parameters
    ----------
    df:
        Transaction DataFrame with ``timestamp`` and ``is_fraud`` columns.
    freq:
        Pandas frequency string for resampling (e.g. ``"D"`` daily,
        ``"W"`` weekly, ``"H"`` hourly).

    Returns
    -------
    pd.DataFrame
        Columns: ``date``, ``total``, ``fraud_count``, ``fraud_rate``.
    """
    tmp = df.copy()
    tmp["timestamp"] = pd.to_datetime(tmp["timestamp"])
    tmp = tmp.set_index("timestamp")

    grouped = tmp.resample(freq).agg(
        total=("is_fraud", "count"),
        fraud_count=("is_fraud", "sum"),
    ).reset_index()
    grouped["fraud_rate"] = grouped["fraud_count"] / grouped["total"].replace(0, np.nan)
    grouped = grouped.rename(columns={"timestamp": "date"})
    return grouped.dropna(subset=["fraud_rate"])


# ---------------------------------------------------------------------------
# High-risk account table
# ---------------------------------------------------------------------------

@timed
def compute_high_risk_accounts(
    df: pd.DataFrame,
    top_n: int = 20,
) -> pd.DataFrame:
    """Build a ranked table of the highest-risk sender accounts.

    Parameters
    ----------
    df:
        Transaction DataFrame with ``sender_account``, ``is_fraud``,
        ``amount``, and optionally ``fraud_probability`` columns.
    top_n:
        Maximum number of accounts to return.

    Returns
    -------
    pd.DataFrame
        Columns: ``account``, ``total_txns``, ``fraud_txns``,
        ``fraud_rate``, ``total_amount``, ``avg_fraud_prob``.
    """
    agg: dict = {
        "total_txns": ("is_fraud", "count"),
        "fraud_txns": ("is_fraud", "sum"),
        "total_amount": ("amount", "sum"),
    }
    if "fraud_probability" in df.columns:
        agg["avg_fraud_prob"] = ("fraud_probability", "mean")

    table = (
        df.groupby("sender_account")
        .agg(**agg)
        .reset_index()
        .rename(columns={"sender_account": "account"})
    )
    table["fraud_rate"] = table["fraud_txns"] / table["total_txns"]
    table = table.sort_values("fraud_rate", ascending=False).head(top_n)
    return table.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Anomaly summary
# ---------------------------------------------------------------------------

@timed
def compute_anomaly_summary(
    df: pd.DataFrame,
    anomaly_scores: np.ndarray,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Attach anomaly scores to the transaction DataFrame and flag outliers.

    Parameters
    ----------
    df:
        Transaction DataFrame.
    anomaly_scores:
        Array of scores in [0, 1] (output of
        :meth:`models.anomaly_detector.AnomalyDetector.anomaly_scores`).
    threshold:
        Score above which a transaction is considered anomalous.

    Returns
    -------
    pd.DataFrame
        Original DataFrame with two new columns: ``anomaly_score``
        and ``is_anomaly``.
    """
    out = df.copy()
    out["anomaly_score"] = anomaly_scores
    out["is_anomaly"] = (anomaly_scores >= threshold).astype(int)
    return out


# ---------------------------------------------------------------------------
# Category breakdown
# ---------------------------------------------------------------------------

def compute_category_fraud_rates(df: pd.DataFrame) -> pd.DataFrame:
    """Compute fraud rate per merchant category.

    Returns
    -------
    pd.DataFrame
        Columns: ``merchant_category``, ``total``, ``fraud_count``, ``fraud_rate``.
    """
    table = (
        df.groupby("merchant_category")
        .agg(total=("is_fraud", "count"), fraud_count=("is_fraud", "sum"))
        .reset_index()
    )
    table["fraud_rate"] = table["fraud_count"] / table["total"]
    return table.sort_values("fraud_rate", ascending=False)
