"""
config.py — Centralized configuration for FraudNet AI.

All tuneable parameters live here so every module can import from a single
source of truth instead of hard-coding values inline.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Data generation defaults
# ---------------------------------------------------------------------------
DEFAULT_N_TRANSACTIONS: int = 10_000
DEFAULT_FRAUD_RATE: float = 0.05          # 5 % fraudulent transactions
DEFAULT_RANDOM_SEED: int = 42

ACCOUNT_POOL_SIZE: int = 500             # unique account IDs in the simulation
MERCHANT_CATEGORIES: list[str] = [
    "grocery", "electronics", "travel", "entertainment",
    "restaurant", "healthcare", "online_retail", "atm_withdrawal",
    "cryptocurrency", "gambling",
]
DEVICE_TYPES: list[str] = ["mobile", "desktop", "tablet", "pos_terminal"]
LOCATIONS: list[str] = [
    "New York", "Los Angeles", "Chicago", "Houston", "Phoenix",
    "Philadelphia", "San Antonio", "San Diego", "Dallas", "San Jose",
    "London", "Tokyo", "Berlin", "Sydney", "Dubai",
]

# ---------------------------------------------------------------------------
# Model hyper-parameters
# ---------------------------------------------------------------------------
RANDOM_FOREST_PARAMS: dict = {
    "n_estimators": 200,
    "max_depth": 12,
    "min_samples_split": 5,
    "min_samples_leaf": 2,
    "class_weight": "balanced",
    "random_state": DEFAULT_RANDOM_SEED,
    "n_jobs": -1,
}

ISOLATION_FOREST_PARAMS: dict = {
    "n_estimators": 150,
    "contamination": 0.05,
    "random_state": DEFAULT_RANDOM_SEED,
    "n_jobs": -1,
}

TEST_SIZE: float = 0.20
CROSS_VAL_FOLDS: int = 5

# ---------------------------------------------------------------------------
# Network / graph settings
# ---------------------------------------------------------------------------
FRAUD_RING_MIN_CYCLE_LENGTH: int = 3     # minimum hops to flag a cycle
FRAUD_RING_MAX_CYCLE_LENGTH: int = 6
HIGH_RISK_DEGREE_THRESHOLD: int = 10     # nodes with degree above this are flagged

# ---------------------------------------------------------------------------
# UI / dashboard settings
# ---------------------------------------------------------------------------
DASHBOARD_TITLE: str = "FraudNet AI — Financial Fraud Detection Platform"
PLOTLY_TEMPLATE: str = "plotly_dark"
COLOR_NORMAL: str = "#4A90D9"
COLOR_FRAUD: str = "#E74C3C"
COLOR_WARNING: str = "#F39C12"
COLOR_SUCCESS: str = "#2ECC71"
