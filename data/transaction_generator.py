"""
data.transaction_generator
===========================
Generates synthetic financial transaction data with configurable fraud rate.

The generator produces a :class:`pandas.DataFrame` that closely mimics real
payment-network data, with engineered features ready for the downstream ML
pipeline.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd

import config


class TransactionGenerator:
    """Generate synthetic financial transaction data.

    Parameters
    ----------
    n_transactions:
        Total number of transactions to generate.
    fraud_rate:
        Fraction of transactions that are labelled as fraudulent (0–1).
    random_seed:
        Seed for reproducibility.
    """

    def __init__(
        self,
        n_transactions: int = config.DEFAULT_N_TRANSACTIONS,
        fraud_rate: float = config.DEFAULT_FRAUD_RATE,
        random_seed: int = config.DEFAULT_RANDOM_SEED,
    ) -> None:
        self.n_transactions = n_transactions
        self.fraud_rate = fraud_rate
        self.random_seed = random_seed
        self._rng = np.random.default_rng(random_seed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self) -> pd.DataFrame:
        """Generate and return the full transaction dataset.

        Returns
        -------
        pd.DataFrame
            One row per transaction with raw fields **and** engineered
            features.
        """
        n_fraud = max(1, int(self.n_transactions * self.fraud_rate))
        n_normal = self.n_transactions - n_fraud

        normal_df = self._build_transactions(n_normal, is_fraud=False)
        fraud_df = self._build_transactions(n_fraud, is_fraud=True)

        df = (
            pd.concat([normal_df, fraud_df], ignore_index=True)
            .sample(frac=1, random_state=self.random_seed)
            .reset_index(drop=True)
        )

        df = self._engineer_features(df)
        return df

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_transactions(self, n: int, is_fraud: bool) -> pd.DataFrame:
        """Build *n* raw transaction rows."""
        # Account pools
        if is_fraud:
            # Fraud transactions reuse a small pool of accounts to create
            # detectable patterns (velocity, rings, …)
            n_accounts = max(10, n // 5)
            senders = [f"ACC_{i:05d}" for i in self._rng.integers(0, n_accounts, n)]
            receivers = [f"ACC_{i:05d}" for i in self._rng.integers(0, n_accounts, n)]
        else:
            n_accounts = max(100, n // 2)
            senders = [f"ACC_{i:05d}" for i in self._rng.integers(0, n_accounts, n)]
            receivers = [f"ACC_{i:05d}" for i in self._rng.integers(0, n_accounts, n)]

        # Amounts
        if is_fraud:
            amounts = np.abs(
                self._rng.normal(config.AMOUNT_MEAN_FRAUD, config.AMOUNT_STD_FRAUD, n)
            ).round(2)
        else:
            amounts = np.abs(
                self._rng.normal(config.AMOUNT_MEAN_NORMAL, config.AMOUNT_STD_NORMAL, n)
            ).round(2)

        # Timestamps — spread across 90 days ending now
        base_ts = datetime(2024, 1, 1)
        seconds_offsets = self._rng.integers(0, 90 * 24 * 3600, n)
        timestamps = [base_ts + timedelta(seconds=int(s)) for s in seconds_offsets]

        merchant_categories = self._rng.choice(config.MERCHANT_CATEGORIES, n)
        device_types = self._rng.choice(config.DEVICE_TYPES, n)
        locations = self._rng.choice(config.LOCATIONS, n)

        return pd.DataFrame(
            {
                "transaction_id": [str(uuid.uuid4()) for _ in range(n)],
                "sender_account": senders,
                "receiver_account": receivers,
                "amount": amounts,
                "merchant_category": merchant_categories,
                "device_type": device_types,
                "location": locations,
                "timestamp": timestamps,
                "is_fraud": int(is_fraud),
            }
        )

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Derive ML-ready numeric features from raw transaction fields."""
        df = df.copy()

        # transaction_amount  (direct)
        df["transaction_amount"] = df["amount"]

        # time_of_day  (hour, 0–23)
        df["time_of_day"] = pd.to_datetime(df["timestamp"]).dt.hour

        # merchant_risk  (lookup from config)
        df["merchant_risk"] = df["merchant_category"].map(config.MERCHANT_RISK_MAP).fillna(0.3)

        # transaction_frequency  — number of transactions sent by this account
        #   in the whole dataset (proxy for velocity)
        freq = df.groupby("sender_account")["transaction_id"].transform("count")
        df["transaction_frequency"] = freq

        # location_change  — 1 if a sender uses more than one location
        loc_nunique = df.groupby("sender_account")["location"].transform("nunique")
        df["location_change"] = (loc_nunique > 1).astype(int)

        # device_change  — 1 if a sender uses more than one device type
        dev_nunique = df.groupby("sender_account")["device_type"].transform("nunique")
        df["device_change"] = (dev_nunique > 1).astype(int)

        return df
