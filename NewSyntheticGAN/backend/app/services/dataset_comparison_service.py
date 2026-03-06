"""
Dataset Comparison Service
==========================
Compares an original financial dataset against a synthetic one using:
  - Per-column distribution similarity (KS test)
  - Correlation matrix similarity (Frobenius norm)
  - Fraud pattern preservation check
"""

import logging
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

logger = logging.getLogger(__name__)


def compare_datasets(
    original_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    original_id: int,
    synthetic_id: int,
) -> dict[str, Any]:
    """
    Produce a statistical similarity report comparing *original_df* and
    *synthetic_df*.

    Returns a dict conforming to ``ComparisonReport`` schema.
    """
    logger.info(
        "Comparing dataset %d (%d rows) vs dataset %d (%d rows)",
        original_id, len(original_df), synthetic_id, len(synthetic_df),
    )

    # ── Shared numeric columns ──────────────────────────────────────────
    orig_num = original_df.select_dtypes(include=[np.number])
    syn_num = synthetic_df.select_dtypes(include=[np.number])
    shared_cols = sorted(set(orig_num.columns) & set(syn_num.columns))

    # ── Per-column KS test ──────────────────────────────────────────────
    column_similarity: dict[str, float] = {}
    for col in shared_cols:
        stat, _ = ks_2samp(
            orig_num[col].dropna().values,
            syn_num[col].dropna().values,
        )
        # Similarity = 1 − KS statistic  (1 = identical distributions)
        column_similarity[col] = round(1.0 - float(stat), 4)

    overall_score = (
        round(float(np.mean(list(column_similarity.values()))), 4)
        if column_similarity
        else 0.0
    )

    # ── Correlation similarity ──────────────────────────────────────────
    if len(shared_cols) >= 2:
        orig_corr = orig_num[shared_cols].corr().values
        syn_corr = syn_num[shared_cols].corr().values
        # Normalised Frobenius distance → similarity
        frob = np.linalg.norm(orig_corr - syn_corr, "fro")
        max_frob = np.sqrt(len(shared_cols) ** 2 * 4)  # theoretical max
        correlation_similarity = round(1.0 - float(frob / max_frob), 4)
    else:
        correlation_similarity = 1.0  # trivially similar

    # ── Fraud distribution ──────────────────────────────────────────────
    fraud_col_orig = _find_fraud_column(original_df)
    fraud_col_syn = _find_fraud_column(synthetic_df)

    fraud_ratio_orig = (
        round(float(original_df[fraud_col_orig].mean()), 4) if fraud_col_orig else 0.0
    )
    fraud_ratio_syn = (
        round(float(synthetic_df[fraud_col_syn].mean()), 4) if fraud_col_syn else 0.0
    )

    # Consider fraud preserved if difference < 5 percentage points
    fraud_preserved = abs(fraud_ratio_orig - fraud_ratio_syn) < 0.05

    report = {
        "original_dataset_id": original_id,
        "synthetic_dataset_id": synthetic_id,
        "column_distribution_similarity": column_similarity,
        "overall_distribution_score": overall_score,
        "correlation_similarity": correlation_similarity,
        "fraud_ratio_original": fraud_ratio_orig,
        "fraud_ratio_synthetic": fraud_ratio_syn,
        "fraud_distribution_preserved": fraud_preserved,
    }

    logger.info(
        "Comparison done -- overall dist score: %.4f | corr similarity: %.4f",
        overall_score, correlation_similarity,
    )
    return report


# ═══════════════════════════════════════════════════════════════════════════
# Internal Helpers
# ═══════════════════════════════════════════════════════════════════════════

_FRAUD_ALIASES = {"fraudlabel", "fraud", "is_fraud", "isfraud", "label", "fraud_flag"}


def _find_fraud_column(df: pd.DataFrame) -> str | None:
    for col in df.columns:
        if col.lower().replace(" ", "").replace("_", "") in {
            a.replace("_", "") for a in _FRAUD_ALIASES
        }:
            return col
    return None
