# FraudNet AI — Financial Fraud Detection & Transaction Intelligence Platform

A production-quality AI system for simulating financial transactions and detecting fraudulent behaviour using Machine Learning, Anomaly Detection, and Graph Intelligence.

---

## Features

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Transaction Generator | NumPy / Pandas | Synthetic financial data (10 000+ transactions) |
| Fraud Classifier | RandomForestClassifier | Supervised fraud probability scoring |
| Anomaly Detector | IsolationForest | Unsupervised abnormal-behaviour detection |
| Transaction Graph | NetworkX | Account relationship modelling |
| Fraud Ring Detector | NetworkX cycle detection | Circular money-transfer pattern identification |
| Dashboard | Streamlit + Plotly | Interactive real-time analytics UI |

---

## Project Structure

```
fraudnet-ai/
├── app.py                        # Streamlit entry point
├── config.py                     # Centralised configuration
├── requirements.txt
├── data/
│   └── transaction_generator.py  # Synthetic data generation
├── models/
│   ├── fraud_classifier.py       # RandomForest fraud detector
│   └── anomaly_detector.py       # IsolationForest anomaly detector
├── network/
│   ├── transaction_graph.py      # NetworkX graph builder
│   └── fraud_ring_detector.py    # Cycle / fraud-ring detector
├── analytics/
│   └── fraud_metrics.py          # KPI & trend computations
├── ui/
│   ├── dashboard.py              # Streamlit dashboard sections
│   └── charts.py                 # Plotly chart factory
└── utils/
    └── helpers.py                # Shared utility functions
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the app

```bash
streamlit run app.py
```

Then open the URL printed in your terminal (usually http://localhost:8501).

---

## Dashboard Sections

### 📊 Fraud Overview
KPI cards: Total Transactions · Fraudulent Transactions · Fraud Rate · High-Risk Accounts

### 🤖 Detection Analytics
- Fraud probability distribution histogram
- Confusion matrix heat-map
- Feature importance bar chart
- Full classification report

### 🕸️ Transaction Network
- Interactive NetworkX graph (nodes coloured by risk score)
- Fraud ring detection results table

### 📈 Risk Monitoring
- Fraud trend over time (dual-axis chart)
- Top high-risk accounts
- Transaction anomaly scatter plot
- Fraud rate by merchant category

---

## Simulation Controls

Use the sidebar to configure:
- **Number of Transactions** (1 000 – 50 000)
- **Fraud Percentage** (1 % – 30 %)
- **Random Seed** (for reproducibility)

Click **🚀 Run Simulation** to generate data and train all models.

---

## Deployment on Streamlit Cloud

1. Push this repository to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io) and create a new app.
3. Set **Main file path** to `app.py`.
4. Click **Deploy** — Streamlit Cloud will install `requirements.txt` automatically.

---

## Tech Stack

- **Python 3.10+**
- **Streamlit** — interactive web UI
- **scikit-learn** — RandomForest & IsolationForest
- **NetworkX** — graph construction and cycle detection
- **Plotly** — interactive charts
- **Pandas / NumPy** — data manipulation
