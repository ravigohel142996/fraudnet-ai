"""
app.py — FraudNet AI main entry point.

Run with:
    streamlit run app.py

All session state is managed here; the dashboard and model modules are
imported as pure workers with no Streamlit coupling of their own.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import streamlit as st

# Ensure the project root is on the path when running via `streamlit run`
ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analytics.fraud_metrics import compute_anomaly_summary, compute_kpis
from config import DASHBOARD_TITLE, DEFAULT_FRAUD_RATE, DEFAULT_N_TRANSACTIONS, DEFAULT_RANDOM_SEED
from data.transaction_generator import generate_transactions
from models.anomaly_detector import AnomalyDetector
from models.fraud_classifier import FraudClassifier
from network.fraud_ring_detector import FraudRingDetector
from network.transaction_graph import TransactionGraph
from ui.dashboard import (
    render_ml_analytics,
    render_network,
    render_overview,
    render_risk_monitoring,
    render_sidebar,
)
from utils.helpers import configure_logging

configure_logging(logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Streamlit page config (must be called first)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title=DASHBOARD_TITLE,
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .stMetric { background-color: #1e2130; border-radius: 8px; padding: 8px; }
    .stMetric label { font-size: 0.85rem; color: #a0aec0; }
    .block-container { padding-top: 1.5rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# App header
# ---------------------------------------------------------------------------
st.title("🔍 " + DASHBOARD_TITLE)
st.markdown(
    "Real-time fraud detection using **Machine Learning**, **Anomaly Detection**, "
    "and **Graph Intelligence**."
)
st.markdown("---")


# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------
params = render_sidebar()


# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------
if "simulation_run" not in st.session_state:
    st.session_state.simulation_run = False
    st.session_state.df = None
    st.session_state.classifier = None
    st.session_state.anomaly_detector = None
    st.session_state.txn_graph = None
    st.session_state.ring_detector = None
    st.session_state.kpis = None


# ---------------------------------------------------------------------------
# Simulation runner
# ---------------------------------------------------------------------------
def run_simulation(n_transactions: int, fraud_rate: float, random_seed: int) -> None:
    """Execute the full simulation pipeline and cache results in session state."""

    progress = st.progress(0, text="Generating transactions…")

    # 1. Generate data
    df = generate_transactions(n_transactions, fraud_rate, random_seed)
    progress.progress(20, text="Training fraud classifier…")

    # 2. Fraud classifier
    classifier = FraudClassifier(random_seed=random_seed)
    classifier.fit(df)
    df["fraud_probability"] = classifier.predict_proba(df)
    progress.progress(45, text="Running anomaly detection…")

    # 3. Anomaly detector
    detector = AnomalyDetector(random_seed=random_seed)
    detector.fit(df)
    anomaly_scores = detector.anomaly_scores(df)
    df = compute_anomaly_summary(df, anomaly_scores)
    progress.progress(65, text="Building transaction graph…")

    # 4. Transaction graph
    txn_graph = TransactionGraph(df)
    progress.progress(80, text="Detecting fraud rings…")

    # 5. Fraud ring detection
    ring_detector = FraudRingDetector(txn_graph)
    ring_detector.detect()
    progress.progress(95, text="Computing KPIs…")

    # 6. KPIs
    kpis = compute_kpis(df)
    progress.progress(100, text="Done!")
    progress.empty()

    # Cache everything in session state
    st.session_state.df = df
    st.session_state.classifier = classifier
    st.session_state.anomaly_detector = detector
    st.session_state.txn_graph = txn_graph
    st.session_state.ring_detector = ring_detector
    st.session_state.kpis = kpis
    st.session_state.simulation_run = True

    logger.info("Simulation complete (%d transactions)", n_transactions)


# ---------------------------------------------------------------------------
# Trigger simulation
# ---------------------------------------------------------------------------
if params["run_simulation"]:
    with st.spinner("Running FraudNet AI simulation…"):
        run_simulation(
            params["n_transactions"],
            params["fraud_rate"],
            params["random_seed"],
        )
    st.success("✅ Simulation complete!")

# ---------------------------------------------------------------------------
# Dashboard rendering
# ---------------------------------------------------------------------------
if st.session_state.simulation_run:
    df = st.session_state.df
    classifier = st.session_state.classifier
    anomaly_detector = st.session_state.anomaly_detector
    txn_graph = st.session_state.txn_graph
    ring_detector = st.session_state.ring_detector
    kpis = st.session_state.kpis

    # Tab layout
    tab_overview, tab_ml, tab_network, tab_risk = st.tabs([
        "📊 Fraud Overview",
        "🤖 Detection Analytics",
        "🕸️ Transaction Network",
        "📈 Risk Monitoring",
    ])

    with tab_overview:
        render_overview(kpis)

    with tab_ml:
        render_ml_analytics(df, classifier)

    with tab_network:
        render_network(txn_graph, ring_detector)

    with tab_risk:
        render_risk_monitoring(df, anomaly_detector)

else:
    # Welcome screen shown before first simulation run
    st.info(
        "👈 **Configure simulation parameters in the sidebar and click 'Run Simulation'** to begin."
    )
    st.markdown(
        """
        ### How it works

        | Step | Component | Description |
        |------|-----------|-------------|
        | 1 | **Transaction Generator** | Creates synthetic financial transactions with fraud labels |
        | 2 | **Fraud Classifier** | RandomForest model predicts fraud probability |
        | 3 | **Anomaly Detector** | IsolationForest flags unusual spending patterns |
        | 4 | **Transaction Graph** | NetworkX graph reveals account relationships |
        | 5 | **Fraud Ring Detector** | Identifies circular money-transfer schemes |
        """
    )
