"""
models/anomaly_detector.py — IsolationForest-based anomaly detection.

Complements the supervised FraudClassifier by flagging transactions that
look statistically unusual regardless of their fraud label.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler

from config import DEFAULT_RANDOM_SEED, ISOLATION_FOREST_PARAMS
from utils.helpers import timed

logger = logging.getLogger(__name__)

ANOMALY_FEATURE_COLUMNS: list[str] = [
    "log_amount",
    "transaction_frequency",
    "merchant_risk",
    "time_of_day",
    "location_change",
    "device_change",
    "hour",
]


class AnomalyDetector:
    """IsolationForest anomaly detector for financial transactions.

    Generates an *anomaly score* in [0, 1] where higher values indicate
    more unusual behaviour.  A score ≥ 0.5 is treated as an anomaly.

    Usage::

        det = AnomalyDetector()
        det.fit(transactions_df)
        scores = det.anomaly_scores(transactions_df)
    """

    def __init__(self, random_seed: int = DEFAULT_RANDOM_SEED) -> None:
        self.random_seed = random_seed
        self._model: Optional[IsolationForest] = None
        self._scaler: RobustScaler = RobustScaler()
        self.metrics_: dict = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @timed
    def fit(self, df: pd.DataFrame) -> "AnomalyDetector":
        """Fit the IsolationForest on *df*.

        Parameters
        ----------
        df:
            DataFrame returned by :func:`data.transaction_generator.generate_transactions`.

        Returns
        -------
        self
        """
        X = self._extract_features(df)
        X_scaled = self._scaler.fit_transform(X)

        params = {**ISOLATION_FOREST_PARAMS, "random_state": self.random_seed}
        self._model = IsolationForest(**params)
        self._model.fit(X_scaled)

        # Compute training-set statistics for monitoring
        raw_scores = self._model.score_samples(X_scaled)
        self.metrics_ = {
            "mean_anomaly_score": float(np.mean(raw_scores)),
            "std_anomaly_score": float(np.std(raw_scores)),
            "anomaly_count": int((raw_scores < 0).sum()),
        }
        logger.info(
            "AnomalyDetector fitted | anomalies_detected=%d / %d",
            self.metrics_["anomaly_count"],
            len(df),
        )
        return self

    def anomaly_scores(self, df: pd.DataFrame) -> np.ndarray:
        """Return normalised anomaly scores in [0, 1] for each row.

        Scores closer to 1 indicate more anomalous transactions.
        """
        self._check_fitted()
        X = self._extract_features(df)
        X_scaled = self._scaler.transform(X)
        raw = self._model.score_samples(X_scaled)  # type: ignore[union-attr]
        # score_samples returns negative values; more negative = more anomalous
        # Normalise to [0, 1]: 1 = most anomalous
        normed = (raw - raw.min()) / (raw.max() - raw.min() + 1e-9)
        return 1.0 - normed  # invert so 1 = most anomalous

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Return binary anomaly flags (1 = anomaly, 0 = normal)."""
        self._check_fitted()
        X = self._extract_features(df)
        X_scaled = self._scaler.transform(X)
        labels = self._model.predict(X_scaled)  # type: ignore[union-attr]
        # IsolationForest returns -1 for anomalies, +1 for normal
        return (labels == -1).astype(int)

    @property
    def is_fitted(self) -> bool:
        """True after :meth:`fit` has been called successfully."""
        return self._model is not None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _extract_features(self, df: pd.DataFrame) -> np.ndarray:
        missing = [c for c in ANOMALY_FEATURE_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"DataFrame missing columns: {missing}")
        return df[ANOMALY_FEATURE_COLUMNS].fillna(0).values.astype(np.float64)

    def _check_fitted(self) -> None:
        if not self.is_fitted:
            raise RuntimeError("Call fit() before predict()")
