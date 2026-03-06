# 🔍 FraudNet AI — Financial Fraud Detection & Transaction Intelligence

FraudNet AI is a production-quality AI system that simulates financial
transactions and detects fraudulent behaviour using Machine Learning, Anomaly
Detection, and Graph Intelligence.

---

## ✨ Features

| Component | Technology |
|-----------|-----------|
| Fraud Classification | Random Forest (`scikit-learn`) |
| Anomaly Detection | Isolation Forest (`scikit-learn`) |
| Transaction Graph | Directed graph (`NetworkX`) |
| Fraud Ring Detection | Directed cycle enumeration (`NetworkX`) |
| Interactive Dashboard | `Streamlit` + `Plotly` |

---

## 🗂 Project Structure

```
fraudnet-ai/
├── app.py                         # Streamlit entry-point
├── config.py                      # All tuneable constants
├── requirements.txt
│
├── data/
│   └── transaction_generator.py   # Synthetic transaction data generator
│
├── models/
│   ├── fraud_classifier.py        # Random Forest fraud classifier
│   └── anomaly_detector.py        # Isolation Forest anomaly detector
│
├── network/
│   ├── transaction_graph.py       # NetworkX directed graph
│   └── fraud_ring_detector.py     # Circular-pattern (ring) detection
│
├── analytics/
│   └── fraud_metrics.py           # KPIs, trend analysis, risk ranking
│
├── ui/
│   ├── dashboard.py               # Streamlit layout & section renderers
│   └── charts.py                  # Plotly figure factories
│
└── utils/
    └── helpers.py                 # Shared stateless utilities
```

---

## 🚀 Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the dashboard

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`.

### 3. Use the simulation controls

1. Adjust **Number of Transactions**, **Fraud Percentage**, and **Random Seed**
   in the left sidebar.
2. Click **🚀 Run Simulation**.
3. Explore the four dashboard tabs:
   - **📊 Fraud Overview** — KPI cards, probability distribution, trend chart
   - **🤖 Detection Analytics** — confusion matrix, feature importance, ROC-AUC
   - **🕸️ Transaction Network** — interactive node-edge graph
   - **⚠️ Risk Monitoring** — high-risk accounts, hourly heatmap, anomaly scatter

---

## ☁️ Streamlit Cloud Deployment

1. Push the repository to GitHub.
2. Log in to [share.streamlit.io](https://share.streamlit.io).
3. Create a new app pointing to `app.py`.
4. Streamlit Cloud automatically installs `requirements.txt`.

---

## 🛠 Tech Stack

- **Python 3.10+**
- **Streamlit** — interactive web dashboard
- **scikit-learn** — Random Forest + Isolation Forest
- **NetworkX** — graph construction and cycle detection
- **Plotly** — interactive visualisations
- **Pandas / NumPy** — data manipulation
