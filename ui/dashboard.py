"""
ui/dashboard.py — Streamlit dashboard layout for FraudNet AI.

Keeps all Streamlit API calls isolated from ML/analytics logic.
The dashboard is driven by the session-level simulation state stored in
``st.session_state``.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import streamlit as st

from analytics.fraud_metrics import (
    compute_anomaly_summary,
    compute_category_fraud_rates,
    compute_fraud_trend,
    compute_high_risk_accounts,
    compute_kpis,
)
from config import COLOR_FRAUD, COLOR_NORMAL, COLOR_SUCCESS, COLOR_WARNING, DASHBOARD_TITLE
from models.anomaly_detector import AnomalyDetector
from models.fraud_classifier import FraudClassifier
from network.fraud_ring_detector import FraudRingDetector
from network.transaction_graph import TransactionGraph
from ui.charts import (
    anomaly_scatter_chart,
    category_fraud_chart,
    confusion_matrix_heatmap,
    feature_importance_chart,
    fraud_probability_histogram,
    fraud_trend_chart,
    high_risk_accounts_chart,
    network_graph_chart,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def render_sidebar() -> dict:
    """Render the simulation-control sidebar and return the chosen parameters.

    Returns
    -------
    dict
        Keys: ``n_transactions``, ``fraud_rate``, ``random_seed``,
        ``run_simulation``.
    """
    st.sidebar.image(
        "https://img.icons8.com/external-flaticons-lineal-color-flat-icons/64/external-fraud-cyber-security-flaticons-lineal-color-flat-icons.png",
        width=64,
    )
    st.sidebar.title("⚙️ Simulation Controls")

    n_transactions = st.sidebar.slider(
        "Number of Transactions",
        min_value=1_000,
        max_value=50_000,
        value=10_000,
        step=1_000,
        help="Total synthetic transactions to generate",
    )
    fraud_pct = st.sidebar.slider(
        "Fraud Percentage (%)",
        min_value=1,
        max_value=30,
        value=5,
        step=1,
        help="What fraction of transactions are fraudulent",
    )
    random_seed = st.sidebar.number_input(
        "Random Seed",
        min_value=0,
        max_value=9_999,
        value=42,
        step=1,
        help="Seed for reproducible data generation",
    )

    st.sidebar.markdown("---")
    run_simulation = st.sidebar.button("🚀 Run Simulation", use_container_width=True)

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "**FraudNet AI** — Financial Fraud Detection & Transaction Intelligence Platform\n\n"
        "Built with RandomForest · IsolationForest · NetworkX · Plotly"
    )

    return {
        "n_transactions": n_transactions,
        "fraud_rate": fraud_pct / 100.0,
        "random_seed": int(random_seed),
        "run_simulation": run_simulation,
    }


# ---------------------------------------------------------------------------
# Overview section
# ---------------------------------------------------------------------------

def render_overview(kpis: dict) -> None:
    """Render the KPI card row and summary text."""
    st.subheader("📊 Fraud Overview")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Transactions", f"{kpis['total_transactions']:,}")
    col2.metric(
        "Fraudulent Transactions",
        f"{kpis['fraud_count']:,}",
        delta=kpis["fraud_rate_pct"],
        delta_color="inverse",
    )
    col3.metric("Fraud Rate", kpis["fraud_rate_pct"])
    col4.metric("High-Risk Accounts", f"{kpis['high_risk_accounts']:,}")


# ---------------------------------------------------------------------------
# ML analytics section
# ---------------------------------------------------------------------------

def render_ml_analytics(df: pd.DataFrame, classifier: FraudClassifier) -> None:
    """Render fraud probability histogram, confusion matrix, feature importance."""
    st.subheader("🤖 Fraud Detection Analytics")

    col_left, col_right = st.columns([2, 1])
    with col_left:
        st.plotly_chart(
            fraud_probability_histogram(df),
            use_container_width=True,
        )
    with col_right:
        st.plotly_chart(
            confusion_matrix_heatmap(classifier.metrics_["confusion_matrix"]),
            use_container_width=True,
        )

    st.plotly_chart(
        feature_importance_chart(classifier.feature_importance_),
        use_container_width=True,
    )

    # Model metrics table
    report = classifier.metrics_.get("classification_report", {})
    if report:
        metrics_df = pd.DataFrame({
            "Class": ["Legitimate", "Fraud"],
            "Precision": [
                report.get("0", {}).get("precision", 0),
                report.get("1", {}).get("precision", 0),
            ],
            "Recall": [
                report.get("0", {}).get("recall", 0),
                report.get("1", {}).get("recall", 0),
            ],
            "F1-Score": [
                report.get("0", {}).get("f1-score", 0),
                report.get("1", {}).get("f1-score", 0),
            ],
        })
        st.markdown("**Classification Report**")
        st.dataframe(metrics_df.set_index("Class").style.format("{:.3f}"), use_container_width=True)

    acc = classifier.metrics_.get("accuracy", 0)
    auc = classifier.metrics_.get("roc_auc", 0)
    cv_mean = classifier.metrics_.get("cv_roc_auc_mean", 0)
    st.info(
        f"Model Performance — Accuracy: **{acc:.4f}** | ROC-AUC: **{auc:.4f}** "
        f"| CV ROC-AUC: **{cv_mean:.4f}**"
    )


# ---------------------------------------------------------------------------
# Transaction network section
# ---------------------------------------------------------------------------

def render_network(txn_graph: TransactionGraph, ring_detector: FraudRingDetector) -> None:
    """Render the transaction network visualisation and ring-detection results."""
    st.subheader("🕸️ Transaction Network")

    graph_summary = txn_graph.summary()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Network Nodes", f"{graph_summary['nodes']:,}")
    c2.metric("Network Edges", f"{graph_summary['edges']:,}")
    c3.metric("High-Risk Nodes", f"{graph_summary['high_risk_nodes']:,}")
    c4.metric("Avg Risk Score", f"{graph_summary['avg_risk_score']:.3f}")

    st.plotly_chart(
        network_graph_chart(txn_graph.graph, txn_graph.node_risk),
        use_container_width=True,
    )

    # Fraud rings
    ring_summary = ring_detector.summary()
    st.markdown("**Fraud Ring Detection Results**")
    rc1, rc2, rc3 = st.columns(3)
    rc1.metric("Total Rings Detected", ring_summary["total_rings"])
    rc2.metric("High-Risk Rings (≥0.7)", ring_summary["high_risk_rings"])
    rc3.metric("Max Ring Risk Score", f"{ring_summary['max_risk_score']:.3f}")

    if ring_detector.fraud_rings:
        rings_df = pd.DataFrame([
            {
                "Ring Length": r.length,
                "Risk Score": round(r.risk_score, 3),
                "Total Amount ($)": round(r.total_amount, 2),
                "Fraud Edges": r.fraud_edge_count,
                "Path": " → ".join(r.nodes[:4]) + ("…" if r.length > 4 else ""),
            }
            for r in ring_detector.fraud_rings[:20]
        ])
        st.dataframe(rings_df, use_container_width=True)


# ---------------------------------------------------------------------------
# Risk monitoring section
# ---------------------------------------------------------------------------

def render_risk_monitoring(
    df: pd.DataFrame,
    anomaly_detector: AnomalyDetector,
) -> None:
    """Render fraud trend, high-risk accounts, and anomaly scatter."""
    st.subheader("📈 Risk Monitoring")

    # Fraud trend
    trend_df = compute_fraud_trend(df, freq="W")
    st.plotly_chart(fraud_trend_chart(trend_df), use_container_width=True)

    col_l, col_r = st.columns(2)

    # High-risk accounts
    with col_l:
        risk_table = compute_high_risk_accounts(df, top_n=15)
        st.plotly_chart(high_risk_accounts_chart(risk_table), use_container_width=True)

    # Anomaly scatter
    with col_r:
        if "anomaly_score" in df.columns:
            st.plotly_chart(anomaly_scatter_chart(df), use_container_width=True)
        else:
            st.info("Run the simulation to see anomaly scores.")

    # Category fraud rate
    cat_df = compute_category_fraud_rates(df)
    st.plotly_chart(category_fraud_chart(cat_df), use_container_width=True)

    # Raw data explorer
    with st.expander("🔍 Explore Raw Transaction Data"):
        display_cols = [
            "transaction_id", "sender_account", "receiver_account",
            "amount", "merchant_category", "device_type", "location",
            "timestamp", "is_fraud",
        ]
        if "fraud_probability" in df.columns:
            display_cols.append("fraud_probability")
        if "anomaly_score" in df.columns:
            display_cols.append("anomaly_score")
        cols = [c for c in display_cols if c in df.columns]
        st.dataframe(df[cols].head(500), use_container_width=True)
