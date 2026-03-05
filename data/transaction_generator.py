"""
data/transaction_generator.py — Synthetic financial transaction data generation.

Generates realistic-looking financial transaction records with controllable
fraud injection.  All randomness is seeded so results are reproducible.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from config import (
    ACCOUNT_POOL_SIZE,
    DEFAULT_FRAUD_RATE,
    DEFAULT_N_TRANSACTIONS,
    DEFAULT_RANDOM_SEED,
    DEVICE_TYPES,
    LOCATIONS,
    MERCHANT_CATEGORIES,
)
from utils.helpers import generate_account_id, timed

logger = logging.getLogger(__name__)


class TransactionGenerator:
    """Generates synthetic financial transaction datasets.

    Parameters
    ----------
    n_transactions:
        Total number of transactions to generate.
    fraud_rate:
        Fraction of transactions that will be labelled as fraudulent (0–1).
    random_seed:
        Seed for all random-number generators to ensure reproducibility.
    """

    def __init__(
        self,
        n_transactions: int = DEFAULT_N_TRANSACTIONS,
        fraud_rate: float = DEFAULT_FRAUD_RATE,
        random_seed: int = DEFAULT_RANDOM_SEED,
    ) -> None:
        self.n_transactions = n_transactions
        self.fraud_rate = fraud_rate
        self.random_seed = random_seed
        self._rng = np.random.default_rng(random_seed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @timed
    def generate(self) -> pd.DataFrame:
        """Generate the full transaction dataset.

        Returns
        -------
        pd.DataFrame
            One row per transaction, with all raw fields plus derived
            ML features and the ``is_fraud`` ground-truth label.
        """
        logger.info(
            "Generating %d transactions (fraud_rate=%.1f%%, seed=%d)",
            self.n_transactions,
            self.fraud_rate * 100,
            self.random_seed,
        )

        n_fraud = int(self.n_transactions * self.fraud_rate)
        n_legit = self.n_transactions - n_fraud

        legit_df = self._generate_legit_transactions(n_legit)
        fraud_df = self._generate_fraud_transactions(n_fraud)

        df = (
            pd.concat([legit_df, fraud_df], ignore_index=True)
            .sample(frac=1, random_state=self.random_seed)
            .reset_index(drop=True)
        )

        df = self._add_derived_features(df)
        df["transaction_id"] = [f"TXN-{i:07d}" for i in range(len(df))]

        logger.info("Dataset ready: %d rows, %d columns", *df.shape)
        return df

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _account_pool(self, size: int) -> list[str]:
        """Return a list of *size* account IDs from the shared pool."""
        return [generate_account_id(i) for i in range(ACCOUNT_POOL_SIZE)][:size]

    def _generate_legit_transactions(self, n: int) -> pd.DataFrame:
        """Generate *n* legitimate (non-fraud) transactions."""
        rng = self._rng
        pool = self._account_pool(ACCOUNT_POOL_SIZE)

        senders = rng.choice(pool, size=n)
        # Receivers are different from senders
        receivers = np.array([
            rng.choice([a for a in pool if a != s])
            for s in senders
        ])

        timestamps = pd.to_datetime(
            rng.integers(
                int(pd.Timestamp("2024-01-01").timestamp()),
                int(pd.Timestamp("2024-12-31").timestamp()),
                size=n,
            ),
            unit="s",
        )

        return pd.DataFrame({
            "sender_account": senders,
            "receiver_account": receivers,
            "amount": rng.lognormal(mean=4.5, sigma=1.2, size=n).clip(1, 50_000),
            "merchant_category": rng.choice(MERCHANT_CATEGORIES, size=n),
            "device_type": rng.choice(DEVICE_TYPES, size=n),
            "location": rng.choice(LOCATIONS, size=n),
            "timestamp": timestamps,
            "is_fraud": 0,
        })

    def _generate_fraud_transactions(self, n: int) -> pd.DataFrame:
        """Generate *n* fraudulent transactions with characteristic patterns."""
        rng = self._rng
        pool = self._account_pool(ACCOUNT_POOL_SIZE)

        # Fraudsters tend to use a small set of accounts
        fraud_senders = rng.choice(pool[:50], size=n)
        fraud_receivers = rng.choice(pool[50:100], size=n)

        timestamps = pd.to_datetime(
            rng.integers(
                int(pd.Timestamp("2024-01-01").timestamp()),
                int(pd.Timestamp("2024-12-31").timestamp()),
                size=n,
            ),
            unit="s",
        )

        # Fraud transactions: unusually large or suspiciously small (structuring)
        fraud_amounts = np.where(
            rng.random(n) < 0.6,
            rng.lognormal(mean=7.0, sigma=0.8, size=n).clip(1_000, 100_000),  # large
            rng.uniform(1, 9.99, size=n),                                       # structuring
        )

        # High-risk categories dominate fraud
        high_risk_cats = ["cryptocurrency", "gambling", "atm_withdrawal"]
        merchant_categories = np.where(
            rng.random(n) < 0.7,
            rng.choice(high_risk_cats, size=n),
            rng.choice(MERCHANT_CATEGORIES, size=n),
        )

        # Fraud is biased toward off-hours and foreign locations
        foreign_locs = ["London", "Tokyo", "Berlin", "Sydney", "Dubai"]
        locations = np.where(
            rng.random(n) < 0.65,
            rng.choice(foreign_locs, size=n),
            rng.choice(LOCATIONS, size=n),
        )

        return pd.DataFrame({
            "sender_account": fraud_senders,
            "receiver_account": fraud_receivers,
            "amount": fraud_amounts,
            "merchant_category": merchant_categories,
            "device_type": rng.choice(DEVICE_TYPES, size=n),
            "location": locations,
            "timestamp": timestamps,
            "is_fraud": 1,
        })

    def _add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute ML-ready derived features from the raw transaction data."""
        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)

        # Time features
        df["hour"] = df["timestamp"].dt.hour
        df["day_of_week"] = df["timestamp"].dt.dayofweek

        # Transaction frequency per sender in the last 24 h (rolling proxy)
        df = df.sort_values(["sender_account", "timestamp"])
        df["transaction_frequency"] = (
            df.groupby("sender_account")
            .cumcount()
            .clip(upper=50)
        )

        # Location change: 1 if sender's previous transaction was in a different location
        df["prev_location"] = df.groupby("sender_account")["location"].shift(1)
        df["location_change"] = (df["location"] != df["prev_location"]).astype(int)
        df["location_change"] = df["location_change"].fillna(0)

        # Device change
        df["prev_device"] = df.groupby("sender_account")["device_type"].shift(1)
        df["device_change"] = (df["device_type"] != df["prev_device"]).astype(int)
        df["device_change"] = df["device_change"].fillna(0)

        # Merchant risk score
        high_risk = {"cryptocurrency": 1.0, "gambling": 0.9, "atm_withdrawal": 0.7,
                     "online_retail": 0.4, "electronics": 0.3}
        df["merchant_risk"] = df["merchant_category"].map(high_risk).fillna(0.1)

        # Time of day risk (night hours 22-06 are higher risk)
        df["time_of_day"] = df["hour"].apply(
            lambda h: 1.0 if h < 6 or h >= 22 else 0.5 if h < 9 or h >= 20 else 0.0
        )

        # Amount log-transform (stabilises scale for ML)
        df["log_amount"] = np.log1p(df["amount"])

        return df.drop(columns=["prev_location", "prev_device"], errors="ignore")


# ---------------------------------------------------------------------------
# Module-level convenience function
# ---------------------------------------------------------------------------

def generate_transactions(
    n_transactions: int = DEFAULT_N_TRANSACTIONS,
    fraud_rate: float = DEFAULT_FRAUD_RATE,
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> pd.DataFrame:
    """Convenience wrapper around :class:`TransactionGenerator`."""
    return TransactionGenerator(n_transactions, fraud_rate, random_seed).generate()
