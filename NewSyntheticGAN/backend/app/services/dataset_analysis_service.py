"""
Dataset Analysis Service
========================
Analyzes uploaded financial transaction datasets, producing structured reports
covering missing values, duplicates, fraud distribution, outliers, and
suspicious pattern detection.
"""

import logging
from typing import Any

import numpy as np
import pandas as pd

from app.utils.data_cleaning import detect_outliers_iqr

logger = logging.getLogger(__name__)


def analyze_dataset(df: pd.DataFrame, dataset_id: int) -> dict[str, Any]:
    """
    Run a full analysis on *df* and return a structured report dict.

    Parameters
    ----------
    df : pd.DataFrame
        The raw dataset (already loaded from CSV/Excel).
    dataset_id : int
        Database ID of the dataset being analysed.

    Returns
    -------
    dict
        Analysis report conforming to ``DatasetAnalysisReport`` schema.
    """
    logger.info("Analysing dataset %d  (%d rows x %d cols)", dataset_id, *df.shape)

    # ── Basic Info ──────────────────────────────────────────────────────
    row_count, column_count = df.shape
    columns = df.columns.tolist()

    # ── Missing Values ──────────────────────────────────────────────────
    missing_values = df.isnull().sum().to_dict()
    missing_values = {k: int(v) for k, v in missing_values.items()}

    # ── Duplicates ──────────────────────────────────────────────────────
    duplicate_rows = int(df.duplicated().sum())

    # ── Fraud Distribution ──────────────────────────────────────────────
    fraud_col = _find_fraud_column(df)
    if fraud_col:
        fraud_count = int(df[fraud_col].sum())
        normal_count = int(len(df) - fraud_count)
        fraud_ratio = round(fraud_count / max(len(df), 1), 4)
    else:
        fraud_count, normal_count, fraud_ratio = 0, row_count, 0.0

    # ── Amount Distribution ─────────────────────────────────────────────
    amount_col = _find_amount_column(df)
    if amount_col is not None:
        amt = df[amount_col].dropna()
        amount_distribution = {
            "mean": round(float(amt.mean()), 2),
            "median": round(float(amt.median()), 2),
            "std": round(float(amt.std()), 2),
            "min": round(float(amt.min()), 2),
            "max": round(float(amt.max()), 2),
            "q25": round(float(amt.quantile(0.25)), 2),
            "q75": round(float(amt.quantile(0.75)), 2),
        }
    else:
        amount_distribution = _empty_distribution()

    # ── Outliers ────────────────────────────────────────────────────────
    outlier_count = 0
    outlier_transactions: list[dict] = []
    if amount_col is not None:
        mask = detect_outliers_iqr(df[amount_col].dropna())
        outlier_count = int(mask.sum())
        outlier_rows = df.loc[mask.index[mask]].head(20)  # cap at 20
        outlier_transactions = outlier_rows.to_dict(orient="records")
        # JSON-serialise numpy types
        outlier_transactions = _serialise_records(outlier_transactions)

    # ── Suspicious Pattern Detection ────────────────────────────────────
    suspicious_patterns = _detect_suspicious_patterns(df, fraud_col, amount_col)

    report = {
        "dataset_id": dataset_id,
        "row_count": row_count,
        "column_count": column_count,
        "columns": columns,
        "missing_values": missing_values,
        "duplicate_rows": duplicate_rows,
        "fraud_ratio": fraud_ratio,
        "fraud_count": fraud_count,
        "normal_count": normal_count,
        "amount_distribution": amount_distribution,
        "outlier_count": outlier_count,
        "outlier_transactions": outlier_transactions,
        "suspicious_patterns": suspicious_patterns,
    }
    logger.info("Analysis complete for dataset %d", dataset_id)
    return report


# ═══════════════════════════════════════════════════════════════════════════
# Internal Helpers
# ═══════════════════════════════════════════════════════════════════════════

_FRAUD_ALIASES = {"fraudlabel", "fraud", "is_fraud", "isfraud", "label", "fraud_flag"}
_AMOUNT_ALIASES = {"amount", "transactionamount", "transaction_amount", "amt", "value"}


def _find_fraud_column(df: pd.DataFrame) -> str | None:
    for col in df.columns:
        if col.lower().replace(" ", "").replace("_", "") in {
            a.replace("_", "") for a in _FRAUD_ALIASES
        }:
            return col
    return None


def _find_amount_column(df: pd.DataFrame) -> str | None:
    for col in df.columns:
        if col.lower().replace(" ", "").replace("_", "") in {
            a.replace("_", "") for a in _AMOUNT_ALIASES
        }:
            return col
    return None


def _empty_distribution() -> dict:
    return {"mean": 0, "median": 0, "std": 0, "min": 0, "max": 0, "q25": 0, "q75": 0}


def _serialise_records(records: list[dict]) -> list[dict]:
    """Convert numpy types to native Python types for JSON serialisation."""
    clean = []
    for rec in records:
        clean.append(
            {k: (v.item() if isinstance(v, (np.integer, np.floating)) else v) for k, v in rec.items()}
        )
    return clean


def _detect_suspicious_patterns(
    df: pd.DataFrame, fraud_col: str | None, amount_col: str | None
) -> list[str]:
    patterns: list[str] = []

    if fraud_col and amount_col:
        fraud_df = df[df[fraud_col] == 1]
        if len(fraud_df) > 0:
            avg_fraud_amt = fraud_df[amount_col].mean()
            avg_normal_amt = df[df[fraud_col] == 0][amount_col].mean()
            if avg_normal_amt > 0 and avg_fraud_amt > 2 * avg_normal_amt:
                patterns.append(
                    f"Fraud transactions have significantly higher amounts "
                    f"(avg ${avg_fraud_amt:,.2f} vs ${avg_normal_amt:,.2f})"
                )

    if fraud_col:
        fraud_ratio = df[fraud_col].mean()
        if fraud_ratio < 0.01:
            patterns.append(
                f"Severe class imbalance detected -- only {fraud_ratio:.2%} fraud"
            )
        elif fraud_ratio < 0.05:
            patterns.append(
                f"Moderate class imbalance -- {fraud_ratio:.2%} fraud"
            )

    # Check for high-cardinality ID-like columns
    for col in df.select_dtypes(include=["object"]).columns:
        if df[col].nunique() > 0.9 * len(df) and len(df) > 100:
            patterns.append(f"Column '{col}' appears to be a unique identifier")

    if not patterns:
        patterns.append("No obviously suspicious patterns detected")

    return patterns
