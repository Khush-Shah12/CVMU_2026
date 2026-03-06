"""
services/validator_service.py
=============================
Service layer that wraps the AI validator module.

Compares an original dataset with its synthetic counterpart,
producing similarity metrics, fraud comparison, and quality scores.
"""

import logging
import numpy as np
import pandas as pd
from scipy import stats

from ai.validator import validate_dataset
from utils.file_handler import load_dataset
import helpers  # for sanitise_for_json

logger = logging.getLogger(__name__)


def _compute_distribution_similarity(real: pd.Series, synthetic: pd.Series) -> float:
    """
    Compute distribution similarity between two numeric series
    using the Kolmogorov-Smirnov test.
    Returns a similarity score in [0, 1] where 1 = identical distributions.
    """
    try:
        real_clean = real.dropna().astype(float)
        synth_clean = synthetic.dropna().astype(float)
        if len(real_clean) < 2 or len(synth_clean) < 2:
            return 0.0
        ks_stat, _ = stats.ks_2samp(real_clean, synth_clean)
        return round(max(0.0, 1.0 - ks_stat), 4)
    except Exception:
        return 0.0


def _compute_correlation_similarity(real: pd.DataFrame, synthetic: pd.DataFrame) -> float:
    """
    Compare correlation matrices of numeric columns.
    Returns similarity in [0, 1].
    """
    try:
        real_numeric = real.select_dtypes(include=[np.number])
        synth_numeric = synthetic.select_dtypes(include=[np.number])
        common_cols = list(set(real_numeric.columns) & set(synth_numeric.columns))
        if len(common_cols) < 2:
            return 1.0  # can't compare correlations with < 2 columns

        real_corr = real_numeric[common_cols].corr().values.flatten()
        synth_corr = synth_numeric[common_cols].corr().values.flatten()

        # Remove NaN values
        mask = ~(np.isnan(real_corr) | np.isnan(synth_corr))
        if mask.sum() < 2:
            return 1.0

        # Compute correlation of correlations
        similarity = float(np.corrcoef(real_corr[mask], synth_corr[mask])[0, 1])
        return round(max(0.0, similarity), 4)
    except Exception:
        return 0.0


def _compare_fraud_ratio(real: pd.DataFrame, synthetic: pd.DataFrame) -> tuple[str, float, float]:
    """
    Compare fraud ratios between real and synthetic datasets.
    Returns (match_label, real_ratio, synthetic_ratio).
    """
    real_ratio = 0.0
    synth_ratio = 0.0

    if "is_fraud" in real.columns:
        real_ratio = float(real["is_fraud"].mean())
    if "is_fraud" in synthetic.columns:
        synth_ratio = float(synthetic["is_fraud"].mean())

    diff = abs(real_ratio - synth_ratio)
    if diff < 0.02:
        label = "excellent"
    elif diff < 0.05:
        label = "good"
    elif diff < 0.10:
        label = "fair"
    else:
        label = "poor"

    return label, round(real_ratio, 4), round(synth_ratio, 4)


def validate_synthetic_data(dataset_id: str) -> dict:
    """
    Full validation pipeline: compare original vs synthetic dataset.

    Returns
    -------
    dict with similarity_score, fraud_ratio_match, correlation_match,
    quality_score, realism_score, anomaly_score, fraud_patterns_detected,
    and detailed_report.
    """
    # Load both datasets
    original_df = load_dataset(dataset_id, kind="original")
    synthetic_df = load_dataset(dataset_id, kind="synthetic")
    logger.info(f"Validating dataset '{dataset_id}': original={len(original_df)} rows, synthetic={len(synthetic_df)} rows")

    # --- 1. Distribution similarity (on common numeric columns) ---
    common_numeric = list(
        set(original_df.select_dtypes(include=[np.number]).columns)
        & set(synthetic_df.select_dtypes(include=[np.number]).columns)
    )
    dist_scores = {}
    for col in common_numeric:
        dist_scores[col] = _compute_distribution_similarity(original_df[col], synthetic_df[col])
    avg_dist_similarity = round(float(np.mean(list(dist_scores.values()))) if dist_scores else 0.0, 4)

    # --- 2. Correlation similarity ---
    corr_sim = _compute_correlation_similarity(original_df, synthetic_df)
    corr_label = "excellent" if corr_sim > 0.9 else "good" if corr_sim > 0.7 else "fair" if corr_sim > 0.5 else "poor"

    # --- 3. Fraud ratio comparison ---
    fraud_label, real_fraud, synth_fraud = _compare_fraud_ratio(original_df, synthetic_df)

    # --- 4. Run the AI validator on the synthetic dataset ---
    logger.info("Running AI validator on synthetic dataset...")
    ai_validation = validate_dataset(synthetic_df)

    # --- 5. Composite quality score (0 - 10) ---
    quality_score = round(
        (avg_dist_similarity * 3.0 + corr_sim * 3.0 + ai_validation["realism_score"] / 100.0 * 4.0),
        1
    )

    # --- 6. Overall similarity score ---
    similarity_score = round(
        (avg_dist_similarity * 0.4 + corr_sim * 0.3 + ai_validation["realism_score"] / 100.0 * 0.3),
        4
    )

    result = {
        "dataset_id": dataset_id,
        "similarity_score": similarity_score,
        "fraud_ratio_match": fraud_label,
        "correlation_match": corr_label,
        "quality_score": quality_score,
        "realism_score": ai_validation["realism_score"],
        "anomaly_score": ai_validation["anomaly_score"],
        "fraud_patterns_detected": ai_validation["fraud_patterns_detected"],
        "detailed_report": {
            "distribution_similarity": {
                "per_column": dist_scores,
                "average": avg_dist_similarity,
            },
            "correlation_similarity": corr_sim,
            "fraud_comparison": {
                "original_ratio": real_fraud,
                "synthetic_ratio": synth_fraud,
                "match": fraud_label,
            },
            "ai_validation": ai_validation["report"],
        },
    }

    return helpers.sanitise_for_json(result)
