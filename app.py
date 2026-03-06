"""
app.py — FraudNet AI entry-point
=================================
Run with:
    streamlit run app.py
"""

from __future__ import annotations

import streamlit as st

# Page config must be the very first Streamlit call
from ui.dashboard import configure_page

configure_page()

# ---------------------------------------------------------------------------
# Remaining imports (after page config)
# ---------------------------------------------------------------------------
import pandas as pd

from data.transaction_generator import TransactionGenerator
from models.fraud_classifier import FraudClassifier
from models.anomaly_detector import AnomalyDetector
from network.transaction_graph import TransactionGraph
from network.fraud_ring_detector import FraudRingDetector
from analytics import fraud_metrics
from ui import dashboard


# ---------------------------------------------------------------------------
# Session-state helpers
# ---------------------------------------------------------------------------

_STATE_KEYS = ("df", "classifier", "anomaly_detector", "tx_graph", "ring_detector", "kpis")


def _init_state() -> None:
    """Ensure all session-state keys exist."""
    for key in _STATE_KEYS:
        if key not in st.session_state:
            st.session_state[key] = None


def _run_simulation(n_transactions: int, fraud_rate: float, random_seed: int) -> None:
    """Generate data, train models, build graph — store everything in session state."""
    with st.spinner("🔄 Generating transactions…"):
        gen = TransactionGenerator(
            n_transactions=n_transactions,
            fraud_rate=fraud_rate,
            random_seed=random_seed,
        )
        df = gen.generate()

    with st.spinner("🤖 Training fraud classifier…"):
        clf = FraudClassifier(random_state=random_seed)
        clf.fit(df)
        df = clf.annotate(df)

    with st.spinner("🔬 Fitting anomaly detector…"):
        adet = AnomalyDetector(random_state=random_seed)
        adet.fit(df)
        df = adet.annotate(df)

    with st.spinner("🕸️ Building transaction graph…"):
        tx_graph = TransactionGraph(df)

    with st.spinner("🔴 Detecting fraud rings…"):
        ring_det = FraudRingDetector(tx_graph)
        ring_det.detect()

    with st.spinner("📊 Computing KPIs…"):
        kpis = fraud_metrics.compute_kpis(df)

    # Persist in session state
    st.session_state["df"] = df
    st.session_state["classifier"] = clf
    st.session_state["anomaly_detector"] = adet
    st.session_state["tx_graph"] = tx_graph
    st.session_state["ring_detector"] = ring_det
    st.session_state["kpis"] = kpis

    st.success("✅ Simulation complete!")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    _init_state()

    # Sidebar — always rendered
    params = dashboard.render_sidebar()

    # Trigger simulation on button press
    if params["run"]:
        _run_simulation(
            n_transactions=params["n_transactions"],
            fraud_rate=params["fraud_rate"],
            random_seed=params["random_seed"],
        )

    # If no data yet, show welcome screen
    df: pd.DataFrame | None = st.session_state["df"]
    if df is None:
        st.title("🔍 FraudNet AI")
        st.subheader("Financial Fraud Detection & Transaction Intelligence Platform")
        st.info(
            "👈  Use the **Simulation Controls** in the sidebar to generate data, "
            "train the models, and explore the analytics dashboard."
        )
        st.markdown(
            """
            ### What this system does
            | Component | Technology |
            |-----------|-----------|
            | Fraud Classification | Random Forest (scikit-learn) |
            | Anomaly Detection | Isolation Forest (scikit-learn) |
            | Transaction Graph | NetworkX directed graph |
            | Fraud Ring Detection | Directed cycle enumeration (NetworkX) |
            | Visualisation | Plotly interactive charts |
            | Dashboard | Streamlit |
            """
        )
        return

    # ------------------------------------------------------------------ #
    # Dashboard tabs                                                       #
    # ------------------------------------------------------------------ #
    tab1, tab2, tab3, tab4 = st.tabs(
        ["📊 Fraud Overview", "🤖 Detection Analytics", "🕸️ Transaction Network", "⚠️ Risk Monitoring"]
    )

    with tab1:
        dashboard.render_overview(df, st.session_state["kpis"])

    with tab2:
        dashboard.render_detection_analytics(df, st.session_state["classifier"])

    with tab3:
        dashboard.render_network(
            st.session_state["tx_graph"],
            st.session_state["ring_detector"],
        )

    with tab4:
        dashboard.render_risk_monitoring(df)

    # Always show raw data preview at the bottom
    dashboard.render_data_preview(df)


if __name__ == "__main__":
    main()
