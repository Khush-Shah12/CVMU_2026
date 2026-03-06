"""
Metric helper utilities wrapping scikit-learn.
"""

import json
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, Any]:
    """
    Compute standard binary classification metrics.

    Returns dict with: accuracy, precision, recall, f1, confusion_matrix.
    """
    cm = confusion_matrix(y_true, y_pred).tolist()
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "confusion_matrix": cm,
    }


def serialize_confusion_matrix(cm: list[list[int]]) -> str:
    """Serialize a confusion matrix to a JSON string for DB storage."""
    return json.dumps(cm)


def deserialize_confusion_matrix(cm_str: str) -> list[list[int]]:
    """Deserialize a JSON confusion matrix string."""
    return json.loads(cm_str)
