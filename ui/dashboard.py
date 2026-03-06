"""
ui.dashboard
============
Streamlit dashboard layout and rendering for FraudNet AI.

This module is the **only** place where ``streamlit`` is imported.  All ML,
graph and analytics logic is delegated to the respective modules.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd
import streamlit as st

import config
from analytics import fraud_metrics
from ui import charts
from utils.helpers import fmt_currency, fmt_number, fmt_percent, risk_label


# ---------------------------------------------------------------------------
# Page configuration  (called once from app.py)
# ---------------------------------------------------------------------------


def configure_page() -> None:
    """Set Streamlit page config — must be the first Streamlit call."""
    st.set_page_config(
        page_title=config.APP_TITLE,
        page_icon=config.APP_ICON,
        layout="wide",
        initial_sidebar_state="expanded",
    )


# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------


def render_sidebar() -> Dict[str, Any]:
    """Render the simulation-controls sidebar and return parameter dict."""
    with st.sidebar:
        st.title(f"{config.APP_ICON} {config.APP_TITLE}")
        st.caption(config.APP_SUBTITLE)
        st.divider()

        st.subheader("⚙️ Simulation Controls")

        n_transactions = st.slider(
            "Number of Transactions",
            min_value=1_000,
            max_value=50_000,
            value=config.DEFAULT_N_TRANSACTIONS,
            step=1_000,
            help="Total synthetic transactions to generate.",
        )

        fraud_pct = st.slider(
            "Fraud Percentage (%)",
            min_value=1,
            max_value=20,
            value=int(config.DEFAULT_FRAUD_RATE * 100),
            step=1,
            help="Proportion of transactions labelled as fraudulent.",
        )

        random_seed = st.number_input(
            "Random Seed",
            min_value=0,
            max_value=9999,
            value=config.DEFAULT_RANDOM_SEED,
            step=1,
            help="Seed for reproducibility.",
        )

        st.divider()
        run = st.button("🚀 Run Simulation", use_container_width=True, type="primary")

        st.divider()
        st.caption("**Legend**")
        st.markdown(
            f"🔵 Normal transaction  \n🔴 Fraudulent transaction  \n🟡 Fraud-ring account"
        )

    return {
        "n_transactions": n_transactions,
        "fraud_rate": fraud_pct / 100.0,
        "random_seed": int(random_seed),
        "run": run,
    }


# ---------------------------------------------------------------------------
# KPI cards
# ---------------------------------------------------------------------------


def render_kpi_cards(kpis: Dict[str, Any]) -> None:
    """Render the top KPI metric row."""
    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric(
        "Total Transactions",
        fmt_number(kpis["total_transactions"]),
    )
    col2.metric(
        "Fraudulent Transactions",
        fmt_number(kpis["fraudulent_transactions"]),
        delta=f"{kpis['fraud_rate']:.2f}% fraud rate",
        delta_color="inverse",
    )
    col3.metric(
        "Fraud Rate",
        fmt_percent(kpis["fraud_rate"]),
    )
    col4.metric(
        "High-Risk Accounts",
        fmt_number(kpis["high_risk_accounts"]),
    )
    col5.metric(
        "Fraud Amount",
        fmt_currency(kpis["fraud_amount"]),
    )


# ---------------------------------------------------------------------------
# Section: Fraud Overview
# ---------------------------------------------------------------------------


def render_overview(df: pd.DataFrame, kpis: Dict[str, Any]) -> None:
    st.header("📊 Fraud Overview")
    render_kpi_cards(kpis)

    col_left, col_right = st.columns(2)
    with col_left:
        st.plotly_chart(
            charts.fraud_probability_histogram(df),
            use_container_width=True,
        )
    with col_right:
        trend_df = fraud_metrics.fraud_trend(df)
        st.plotly_chart(
            charts.fraud_trend_chart(trend_df),
            use_container_width=True,
        )


# ---------------------------------------------------------------------------
# Section: Fraud Detection Analytics
# ---------------------------------------------------------------------------


def render_detection_analytics(df: pd.DataFrame, classifier: Any) -> None:
    st.header("🤖 Fraud Detection Analytics")

    if classifier is None or not classifier.is_trained:
        st.warning("Classifier not trained yet. Run the simulation first.")
        return

    # ROC-AUC badge
    roc = classifier.roc_auc_ or 0.0
    st.metric("Model ROC-AUC", f"{roc:.4f}")

    col_left, col_mid, col_right = st.columns(3)

    with col_left:
        st.plotly_chart(
            charts.confusion_matrix_heatmap(classifier.confusion_matrix_),
            use_container_width=True,
        )
    with col_mid:
        st.plotly_chart(
            charts.feature_importance_chart(classifier.feature_importances_),
            use_container_width=True,
        )
    with col_right:
        st.subheader("Classification Report")
        st.code(classifier.classification_report_, language=None)


# ---------------------------------------------------------------------------
# Section: Transaction Network
# ---------------------------------------------------------------------------


def render_network(tx_graph: Any, ring_detector: Any) -> None:
    st.header("🕸️ Transaction Network")

    if tx_graph is None:
        st.warning("No graph data. Run the simulation first.")
        return

    summary = tx_graph.summary()
    ring_summary = ring_detector.summary() if ring_detector else {}

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accounts (nodes)", fmt_number(summary["n_nodes"]))
    c2.metric("Transactions (edges)", fmt_number(summary["n_edges"]))
    c3.metric("High-Risk Accounts", fmt_number(summary["n_high_risk"]))
    c4.metric("Fraud Rings Detected", fmt_number(ring_summary.get("n_rings", 0)))

    ring_accts = ring_detector.ring_accounts if ring_detector else set()
    subgraph = tx_graph.top_risk_subgraph(config.MAX_GRAPH_DISPLAY_NODES)
    st.plotly_chart(
        charts.transaction_network_graph(subgraph, ring_accts),
        use_container_width=True,
    )

    if ring_detector and ring_detector.rings:
        with st.expander("🔴 Detected Fraud Rings", expanded=False):
            for i, ring in enumerate(ring_detector.rings[:20], 1):
                st.markdown(f"**Ring {i}:** " + " → ".join(ring) + f" → {ring[0]}")


# ---------------------------------------------------------------------------
# Section: Risk Monitoring
# ---------------------------------------------------------------------------


def render_risk_monitoring(df: pd.DataFrame) -> None:
    st.header("⚠️ Risk Monitoring")

    col_left, col_right = st.columns(2)

    # Top risky accounts
    with col_left:
        risky = fraud_metrics.top_risky_accounts(df, top_n=10)
        if not risky.empty:
            st.plotly_chart(
                charts.high_risk_accounts_chart(risky),
                use_container_width=True,
            )
        else:
            st.info("No risk data available.")

    # Hourly fraud heatmap
    with col_right:
        hourly = fraud_metrics.hourly_fraud_heatmap(df)
        st.plotly_chart(
            charts.hourly_heatmap(hourly),
            use_container_width=True,
        )

    # Anomaly scatter
    if "anomaly_score" in df.columns:
        st.plotly_chart(charts.anomaly_scatter(df), use_container_width=True)


# ---------------------------------------------------------------------------
# Section: Raw Data Preview
# ---------------------------------------------------------------------------


def render_data_preview(df: pd.DataFrame) -> None:
    with st.expander("🗃️ Raw Transaction Data (first 500 rows)", expanded=False):
        display_cols = [
            "transaction_id", "sender_account", "receiver_account",
            "amount", "merchant_category", "device_type", "location",
            "timestamp", "is_fraud",
        ]
        if "fraud_probability" in df.columns:
            display_cols.append("fraud_probability")
        if "is_anomaly" in df.columns:
            display_cols.append("is_anomaly")

        cols_present = [c for c in display_cols if c in df.columns]
        st.dataframe(df[cols_present].head(500), use_container_width=True)
