"""
FraudNet AI — centralised configuration.

All tuneable hyper-parameters, feature lists and default values live here so
that no magic numbers are scattered across the codebase.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Simulation defaults
# ---------------------------------------------------------------------------
DEFAULT_N_TRANSACTIONS: int = 10_000
DEFAULT_FRAUD_RATE: float = 0.05        # 5 %
DEFAULT_RANDOM_SEED: int = 42

# ---------------------------------------------------------------------------
# Transaction generator
# ---------------------------------------------------------------------------
MERCHANT_CATEGORIES: list[str] = [
    "retail", "travel", "dining", "entertainment", "utilities",
    "healthcare", "education", "fuel", "groceries", "online",
]

DEVICE_TYPES: list[str] = ["mobile", "desktop", "tablet", "pos_terminal"]

LOCATIONS: list[str] = [
    "New York", "Los Angeles", "Chicago", "Houston", "Phoenix",
    "Philadelphia", "San Antonio", "San Diego", "Dallas", "San Jose",
    "Austin", "Jacksonville", "Fort Worth", "Columbus", "Charlotte",
    "Indianapolis", "San Francisco", "Seattle", "Denver", "Nashville",
]

# Merchant-category risk scores (0–1)
MERCHANT_RISK_MAP: dict[str, float] = {
    "retail": 0.2,
    "travel": 0.5,
    "dining": 0.15,
    "entertainment": 0.35,
    "utilities": 0.1,
    "healthcare": 0.1,
    "education": 0.05,
    "fuel": 0.2,
    "groceries": 0.1,
    "online": 0.6,
}

AMOUNT_MEAN_NORMAL: float = 150.0
AMOUNT_STD_NORMAL: float = 100.0
AMOUNT_MEAN_FRAUD: float = 800.0
AMOUNT_STD_FRAUD: float = 500.0

# ---------------------------------------------------------------------------
# ML model hyper-parameters
# ---------------------------------------------------------------------------
RF_N_ESTIMATORS: int = 100
RF_MAX_DEPTH: int = 10
RF_MIN_SAMPLES_SPLIT: int = 5
RF_CLASS_WEIGHT: str = "balanced"

ISOLATION_FOREST_CONTAMINATION: float = 0.05
ISOLATION_FOREST_N_ESTIMATORS: int = 100

# ---------------------------------------------------------------------------
# Feature columns used by models
# ---------------------------------------------------------------------------
CLASSIFIER_FEATURES: list[str] = [
    "transaction_amount",
    "transaction_frequency",
    "location_change",
    "device_change",
    "merchant_risk",
    "time_of_day",
]

ANOMALY_FEATURES: list[str] = [
    "transaction_amount",
    "transaction_frequency",
    "merchant_risk",
    "time_of_day",
]

# ---------------------------------------------------------------------------
# Graph / fraud-ring detection
# ---------------------------------------------------------------------------
MAX_GRAPH_DISPLAY_NODES: int = 200
MAX_SUBGRAPH_NODES: int = 80          # cap before nx.simple_cycles (performance)
MIN_RING_LENGTH: int = 3
MAX_RING_LENGTH: int = 8

# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------
APP_TITLE: str = "FraudNet AI"
APP_SUBTITLE: str = "Financial Fraud Detection & Transaction Intelligence"
APP_ICON: str = "🔍"
THEME_FRAUD_COLOR: str = "#EF4444"
THEME_NORMAL_COLOR: str = "#3B82F6"
THEME_WARNING_COLOR: str = "#F59E0B"
THEME_OK_COLOR: str = "#10B981"
