"""
models.anomaly_detector
========================
Unsupervised anomaly detection using Isolation Forest.

Identifies transactions with abnormal spending patterns, unusual frequency
and high-risk behavioural signals — without relying on fraud labels.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

import config


class AnomalyDetector:
    """Isolation-Forest anomaly detector for transaction data.

    Parameters
    ----------
    contamination:
        Expected proportion of anomalous transactions (0–0.5).
    n_estimators:
        Number of isolation trees.
    random_state:
        Seed for reproducibility.
    """

    def __init__(
        self,
        contamination: float = config.ISOLATION_FOREST_CONTAMINATION,
        n_estimators: int = config.ISOLATION_FOREST_N_ESTIMATORS,
        random_state: int = config.DEFAULT_RANDOM_SEED,
    ) -> None:
        self.model = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1,
        )
        self.scaler = StandardScaler()
        self.feature_names: list[str] = config.ANOMALY_FEATURES
        self.is_fitted: bool = False

    # ------------------------------------------------------------------
    # Training / fitting
    # ------------------------------------------------------------------

    def fit(self, df: pd.DataFrame) -> "AnomalyDetector":
        """Fit the detector on *df*.

        Parameters
        ----------
        df:
            DataFrame containing at least the columns in
            ``config.ANOMALY_FEATURES``.

        Returns
        -------
        self
        """
        X = self.scaler.fit_transform(df[self.feature_names].values)
        self.model.fit(X)
        self.is_fitted = True
        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def anomaly_score(self, df: pd.DataFrame) -> np.ndarray:
        """Return a normalised anomaly score in [0, 1] for each row.

        Higher values indicate more anomalous behaviour.
        """
        self._assert_fitted()
        X = self.scaler.transform(df[self.feature_names].values)
        # decision_function returns negative values for anomalies; invert and
        # min-max normalise to [0, 1].
        raw = self.model.decision_function(X)
        normalised = 1.0 - (raw - raw.min()) / (raw.max() - raw.min() + 1e-9)
        return normalised

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Return 1 (anomaly) / 0 (normal) for each row."""
        self._assert_fitted()
        X = self.scaler.transform(df[self.feature_names].values)
        # IsolationForest returns -1 for anomalies, +1 for inliers
        raw = self.model.predict(X)
        return ((raw == -1)).astype(int)

    def annotate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add ``anomaly_score`` and ``is_anomaly`` columns to *df*."""
        out = df.copy()
        out["anomaly_score"] = self.anomaly_score(df)
        out["is_anomaly"] = self.predict(df)
        return out

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _assert_fitted(self) -> None:
        if not self.is_fitted:
            raise RuntimeError("AnomalyDetector must be fitted before inference. Call .fit() first.")
