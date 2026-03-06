"""
Fraud Detection Model Service
==============================
Trains Logistic Regression, Random Forest, and XGBoost classifiers on
financial transaction data and returns per-model evaluation metrics.
"""

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier

from app.utils.metrics import compute_classification_metrics

logger = logging.getLogger(__name__)

# Models to train (name → constructor)
_MODELS: dict[str, Any] = {
    "Logistic Regression": lambda: LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": lambda: RandomForestClassifier(
        n_estimators=100, random_state=42, n_jobs=-1
    ),
    "XGBoost": lambda: XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        use_label_encoder=False,
        eval_metric="logloss",
    ),
}


def train_and_evaluate(
    df: pd.DataFrame,
    label_column: str = "FraudLabel",
    test_size: float = 0.2,
) -> list[dict[str, Any]]:
    """
    Prepare data, train all models, and return evaluation metrics.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset containing features and a binary label column.
    label_column : str
        Name of the target column (0 = normal, 1 = fraud).
    test_size : float
        Fraction of data to hold out for testing.

    Returns
    -------
    list[dict]
        One dict per model with keys: model_name, accuracy, precision,
        recall, f1, confusion_matrix.
    """
    if label_column not in df.columns:
        raise ValueError(f"Label column '{label_column}' not found in dataset")

    logger.info("Preparing data for model training (label=%s)", label_column)

    # ── Feature / label split ───────────────────────────────────────────
    X = df.drop(columns=[label_column])
    y = df[label_column].astype(int)

    # ── Encode categoricals ─────────────────────────────────────────────
    label_encoders: dict[str, LabelEncoder] = {}
    for col in X.select_dtypes(include=["object"]).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le

    # ── Drop remaining non-numeric ──────────────────────────────────────
    X = X.select_dtypes(include=[np.number])

    # ── Handle missing / inf ────────────────────────────────────────────
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0)

    # ── Scale ───────────────────────────────────────────────────────────
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ── Train / test split ──────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=42, stratify=y
    )

    logger.info(
        "Train: %d samples | Test: %d samples | Features: %d",
        len(X_train), len(X_test), X_train.shape[1],
    )

    # ── Train & Evaluate each model ────────────────────────────────────
    results: list[dict[str, Any]] = []
    for name, constructor in _MODELS.items():
        logger.info("Training %s …", name)
        model = constructor()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        metrics = compute_classification_metrics(y_test, y_pred)
        metrics["model_name"] = name
        results.append(metrics)

        logger.info(
            "%s → Accuracy: %.4f | F1: %.4f",
            name, metrics["accuracy"], metrics["f1"],
        )

    return results
