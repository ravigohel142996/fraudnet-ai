"""
analytics.fraud_metrics
========================
Computes portfolio-level fraud KPIs and time-series trends from an annotated
transaction DataFrame.

All functions are pure (no side effects, no Streamlit coupling) and return
plain Python / pandas / numpy objects that the UI layer can render directly.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# KPI aggregates
# ---------------------------------------------------------------------------


def compute_kpis(df: pd.DataFrame) -> Dict[str, float | int | str]:
    """Compute top-level fraud KPIs for the KPI card row.

    Parameters
    ----------
    df:
        Annotated DataFrame with at minimum ``is_fraud``,
        ``fraud_probability``, ``amount``, and ``sender_account`` columns.

    Returns
    -------
    dict with keys:
        ``total_transactions``, ``fraudulent_transactions``, ``fraud_rate``,
        ``total_amount``, ``fraud_amount``, ``avg_fraud_probability``,
        ``high_risk_accounts``.
    """
    total = len(df)
    n_fraud = int(df["is_fraud"].sum())
    fraud_rate = n_fraud / total if total else 0.0

    total_amount = float(df["amount"].sum())
    fraud_amount = float(df.loc[df["is_fraud"] == 1, "amount"].sum())

    avg_fraud_prob = float(df["fraud_probability"].mean()) if "fraud_probability" in df.columns else 0.0

    # High-risk accounts: senders whose mean fraud_probability > 0.5
    if "fraud_probability" in df.columns:
        acct_risk = df.groupby("sender_account")["fraud_probability"].mean()
        high_risk = int((acct_risk > 0.5).sum())
    else:
        high_risk = 0

    return {
        "total_transactions": total,
        "fraudulent_transactions": n_fraud,
        "fraud_rate": round(fraud_rate * 100, 2),          # percent
        "total_amount": round(total_amount, 2),
        "fraud_amount": round(fraud_amount, 2),
        "avg_fraud_probability": round(avg_fraud_prob, 4),
        "high_risk_accounts": high_risk,
    }


# ---------------------------------------------------------------------------
# Time-series analytics
# ---------------------------------------------------------------------------


def fraud_trend(df: pd.DataFrame, freq: str = "D") -> pd.DataFrame:
    """Aggregate fraud counts by time period.

    Parameters
    ----------
    df:
        Annotated DataFrame with ``timestamp`` and ``is_fraud`` columns.
    freq:
        Pandas offset alias for resampling (e.g. ``"D"`` for daily,
        ``"H"`` for hourly).

    Returns
    -------
    pd.DataFrame with columns ``date``, ``total``, ``fraud``, ``fraud_rate``.
    """
    ts = df.copy()
    ts["timestamp"] = pd.to_datetime(ts["timestamp"])
    ts = ts.set_index("timestamp").sort_index()

    total = ts["is_fraud"].resample(freq).count().rename("total")
    fraud_count = ts["is_fraud"].resample(freq).sum().rename("fraud")

    trend = pd.concat([total, fraud_count], axis=1).fillna(0)
    trend["fraud_rate"] = (trend["fraud"] / trend["total"].replace(0, np.nan)).fillna(0)
    trend = trend.reset_index().rename(columns={"timestamp": "date"})
    return trend


def hourly_fraud_heatmap(df: pd.DataFrame) -> pd.DataFrame:
    """Return fraud counts by hour-of-day for a heatmap.

    Returns
    -------
    pd.DataFrame with columns ``hour``, ``total``, ``fraud``.
    """
    ts = df.copy()
    ts["hour"] = pd.to_datetime(ts["timestamp"]).dt.hour
    grouped = ts.groupby("hour").agg(
        total=("is_fraud", "count"),
        fraud=("is_fraud", "sum"),
    ).reset_index()
    grouped["fraud_rate"] = (grouped["fraud"] / grouped["total"].replace(0, np.nan)).fillna(0)
    return grouped


# ---------------------------------------------------------------------------
# Account-level risk ranking
# ---------------------------------------------------------------------------


def top_risky_accounts(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """Return the *top_n* highest-risk sender accounts.

    Returns
    -------
    pd.DataFrame with columns ``sender_account``, ``avg_fraud_probability``,
    ``n_transactions``, ``n_fraud``.
    """
    if "fraud_probability" not in df.columns:
        return pd.DataFrame()

    grouped = df.groupby("sender_account").agg(
        avg_fraud_probability=("fraud_probability", "mean"),
        n_transactions=("transaction_id", "count"),
        n_fraud=("is_fraud", "sum"),
    ).reset_index()

    return grouped.nlargest(top_n, "avg_fraud_probability").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Anomaly breakdown
# ---------------------------------------------------------------------------


def anomaly_breakdown(df: pd.DataFrame) -> Dict[str, int]:
    """Summarise anomaly detection results.

    Returns
    -------
    dict with keys ``total_anomalies``, ``anomaly_overlap_with_fraud``,
    ``anomaly_only``, ``fraud_only``.
    """
    if "is_anomaly" not in df.columns:
        return {}

    total_anomalies = int(df["is_anomaly"].sum())
    overlap = int(((df["is_anomaly"] == 1) & (df["is_fraud"] == 1)).sum())
    anomaly_only = total_anomalies - overlap
    fraud_only = int(df["is_fraud"].sum()) - overlap

    return {
        "total_anomalies": total_anomalies,
        "anomaly_overlap_with_fraud": overlap,
        "anomaly_only": anomaly_only,
        "fraud_only": fraud_only,
    }
