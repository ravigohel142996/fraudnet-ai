"""
models.fraud_classifier
========================
Random-Forest based supervised fraud classifier.

Trained on engineered transaction features and outputs both a binary label
and a continuous *fraud_probability* score.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

import config


class FraudClassifier:
    """Supervised fraud detection using a Random Forest.

    Parameters
    ----------
    n_estimators:   Number of trees.
    max_depth:      Maximum tree depth.
    random_state:   Seed for reproducibility.
    """

    def __init__(
        self,
        n_estimators: int = config.RF_N_ESTIMATORS,
        max_depth: int = config.RF_MAX_DEPTH,
        random_state: int = config.DEFAULT_RANDOM_SEED,
    ) -> None:
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=config.RF_MIN_SAMPLES_SPLIT,
            class_weight=config.RF_CLASS_WEIGHT,
            random_state=random_state,
            n_jobs=-1,
        )
        self.scaler = StandardScaler()
        self.feature_names: list[str] = config.CLASSIFIER_FEATURES
        self.is_trained: bool = False

        # Evaluation artefacts (populated after fit)
        self.confusion_matrix_: Optional[np.ndarray] = None
        self.classification_report_: Optional[str] = None
        self.roc_auc_: Optional[float] = None
        self.feature_importances_: Optional[pd.Series] = None

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, df: pd.DataFrame) -> "FraudClassifier":
        """Train the classifier on *df*.

        Parameters
        ----------
        df:
            DataFrame that must contain the columns listed in
            ``config.CLASSIFIER_FEATURES`` and an ``is_fraud`` column.

        Returns
        -------
        self
        """
        X = df[self.feature_names].values
        y = df["is_fraud"].values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=config.DEFAULT_RANDOM_SEED, stratify=y
        )

        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)

        self.model.fit(X_train, y_train)
        self.is_trained = True

        # Evaluation
        y_pred = self.model.predict(X_test)
        y_prob = self.model.predict_proba(X_test)[:, 1]

        self.confusion_matrix_ = confusion_matrix(y_test, y_pred)
        self.classification_report_ = classification_report(y_test, y_pred)
        self.roc_auc_ = float(roc_auc_score(y_test, y_prob))
        self.feature_importances_ = pd.Series(
            self.model.feature_importances_, index=self.feature_names
        ).sort_values(ascending=False)

        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Return fraud probability for each row in *df*.

        Returns
        -------
        np.ndarray of shape (n,)
        """
        self._assert_trained()
        X = self.scaler.transform(df[self.feature_names].values)
        return self.model.predict_proba(X)[:, 1]

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Return binary fraud prediction (0/1) for each row."""
        return (self.predict_proba(df) >= 0.5).astype(int)

    def annotate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add ``fraud_probability`` and ``predicted_fraud`` columns to *df*."""
        out = df.copy()
        out["fraud_probability"] = self.predict_proba(df)
        out["predicted_fraud"] = self.predict(df)
        return out

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _assert_trained(self) -> None:
        if not self.is_trained:
            raise RuntimeError("FraudClassifier must be trained before inference. Call .fit() first.")
