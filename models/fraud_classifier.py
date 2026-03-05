"""
models/fraud_classifier.py — RandomForest-based fraud detection model.

Encapsulates training, evaluation, and inference in a single clean class so
the dashboard and other consumers never touch raw sklearn APIs.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder

from config import (
    CROSS_VAL_FOLDS,
    DEFAULT_RANDOM_SEED,
    RANDOM_FOREST_PARAMS,
    TEST_SIZE,
)
from utils.helpers import timed

logger = logging.getLogger(__name__)

# Features the model is trained on
FEATURE_COLUMNS: list[str] = [
    "log_amount",
    "transaction_frequency",
    "location_change",
    "device_change",
    "merchant_risk",
    "time_of_day",
    "hour",
    "day_of_week",
]

LABEL_COLUMN: str = "is_fraud"


class FraudClassifier:
    """Random Forest fraud detection model.

    Usage::

        clf = FraudClassifier()
        metrics = clf.fit(transactions_df)
        probs   = clf.predict_proba(new_df)

    Parameters
    ----------
    random_seed:
        Seed passed to the underlying RandomForestClassifier and data split.
    """

    def __init__(self, random_seed: int = DEFAULT_RANDOM_SEED) -> None:
        self.random_seed = random_seed
        self._model: Optional[RandomForestClassifier] = None
        self._label_encoders: dict[str, LabelEncoder] = {}
        self.feature_importance_: Optional[pd.Series] = None
        self.metrics_: dict = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @timed
    def fit(self, df: pd.DataFrame) -> dict:
        """Train the classifier on *df* and return evaluation metrics.

        Parameters
        ----------
        df:
            DataFrame returned by :func:`data.transaction_generator.generate_transactions`.

        Returns
        -------
        dict
            Keys: ``accuracy``, ``roc_auc``, ``cv_roc_auc``,
            ``confusion_matrix``, ``classification_report``.
        """
        X, y = self._prepare_features(df)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=TEST_SIZE,
            random_state=self.random_seed,
            stratify=y,
        )

        self._model = RandomForestClassifier(**RANDOM_FOREST_PARAMS)
        self._model.fit(X_train, y_train)

        # Feature importance
        self.feature_importance_ = pd.Series(
            self._model.feature_importances_,
            index=FEATURE_COLUMNS,
        ).sort_values(ascending=False)

        # Evaluation
        y_pred = self._model.predict(X_test)
        y_proba = self._model.predict_proba(X_test)[:, 1]

        cv = StratifiedKFold(n_splits=CROSS_VAL_FOLDS, shuffle=True, random_state=self.random_seed)
        cv_scores = cross_val_score(self._model, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)

        self.metrics_ = {
            "accuracy": (y_pred == y_test).mean(),
            "roc_auc": roc_auc_score(y_test, y_proba),
            "cv_roc_auc_mean": cv_scores.mean(),
            "cv_roc_auc_std": cv_scores.std(),
            "confusion_matrix": confusion_matrix(y_test, y_pred),
            "classification_report": classification_report(y_test, y_pred, output_dict=True),
        }

        logger.info(
            "FraudClassifier trained | accuracy=%.4f | ROC-AUC=%.4f | CV ROC-AUC=%.4f±%.4f",
            self.metrics_["accuracy"],
            self.metrics_["roc_auc"],
            self.metrics_["cv_roc_auc_mean"],
            self.metrics_["cv_roc_auc_std"],
        )
        return self.metrics_

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Return fraud probability scores for each row in *df*.

        Returns
        -------
        np.ndarray
            1-D array of fraud probabilities in [0, 1].
        """
        self._check_fitted()
        X, _ = self._prepare_features(df, fit_encoders=False)
        return self._model.predict_proba(X)[:, 1]  # type: ignore[union-attr]

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Return binary fraud predictions (0 / 1) for each row in *df*."""
        self._check_fitted()
        X, _ = self._prepare_features(df, fit_encoders=False)
        return self._model.predict(X)  # type: ignore[union-attr]

    @property
    def is_fitted(self) -> bool:
        """True after :meth:`fit` has been called successfully."""
        return self._model is not None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _prepare_features(
        self,
        df: pd.DataFrame,
        fit_encoders: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Extract and encode the feature matrix and label vector."""
        missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"DataFrame is missing columns: {missing}")

        X = df[FEATURE_COLUMNS].copy()
        y = df[LABEL_COLUMN].values if LABEL_COLUMN in df.columns else np.zeros(len(df))

        return X.values.astype(np.float64), y.astype(int)

    def _check_fitted(self) -> None:
        if not self.is_fitted:
            raise RuntimeError("Call fit() before predict()")
