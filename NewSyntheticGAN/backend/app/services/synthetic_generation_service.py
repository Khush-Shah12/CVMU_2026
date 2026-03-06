"""
Synthetic Data Generation Service
==================================
Uses SDV's GaussianCopulaSynthesizer to learn statistical distributions from
a real financial dataset and generate realistic synthetic transactions.
Includes automatic fraud-class balancing via oversampling.
"""

import logging
from typing import Any

import numpy as np
import pandas as pd
from faker import Faker
from sdv.metadata import SingleTableMetadata
from sdv.single_table import GaussianCopulaSynthesizer

logger = logging.getLogger(__name__)
fake = Faker()


def generate_synthetic_data(
    df: pd.DataFrame,
    num_samples: int = 1000,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Generate *num_samples* synthetic rows that mirror the statistical
    properties of *df*.

    Parameters
    ----------
    df : pd.DataFrame
        Original (cleaned) dataset.
    num_samples : int
        Number of synthetic rows to create.

    Returns
    -------
    (synthetic_df, statistics)
        The generated DataFrame and a summary dict.
    """
    logger.info("Starting synthetic generation -- %d samples requested", num_samples)

    # ── Prepare data ────────────────────────────────────────────────────
    work_df = df.copy()

    # Drop unique-ID columns (they'd break the synthesizer or leak PII)
    id_cols = _detect_id_columns(work_df)
    if id_cols:
        logger.info("Dropping ID columns before synthesis: %s", id_cols)
        work_df = work_df.drop(columns=id_cols)

    # ── Build SDV metadata ──────────────────────────────────────────────
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(work_df)

    # ── Fit synthesizer ─────────────────────────────────────────────────
    synthesizer = GaussianCopulaSynthesizer(metadata)
    synthesizer.fit(work_df)

    # ── Sample ──────────────────────────────────────────────────────────
    synthetic_df = synthesizer.sample(num_rows=num_samples)

    # ── Re-add ID column with Faker-generated values ────────────────────
    if id_cols:
        for col in id_cols:
            synthetic_df.insert(
                0, col,
                [fake.uuid4()[:12].upper() for _ in range(len(synthetic_df))],
            )

    # ── Balance fraud class if needed ───────────────────────────────────
    fraud_col = _find_fraud_column(synthetic_df)
    if fraud_col is not None:
        synthetic_df = _balance_fraud(synthetic_df, fraud_col, synthesizer, metadata, id_cols)

    # ── Collect statistics ──────────────────────────────────────────────
    stats = _compute_statistics(synthetic_df, fraud_col)

    logger.info("Synthetic generation complete -- %d rows produced", len(synthetic_df))
    return synthetic_df, stats


def balance_dataset(
    df: pd.DataFrame,
    target_fraud_ratio: float = 0.10,
) -> pd.DataFrame:
    """
    If the fraud ratio in *df* is below *target_fraud_ratio*, generate
    additional synthetic fraud rows and append them.
    """
    fraud_col = _find_fraud_column(df)
    if fraud_col is None:
        logger.warning("No fraud column found -- returning dataset unchanged")
        return df

    current_ratio = df[fraud_col].mean()
    if current_ratio >= target_fraud_ratio:
        logger.info("Fraud ratio %.2f%% already above target -- no balancing needed", current_ratio * 100)
        return df

    # How many extra fraud rows do we need?
    n_total = len(df)
    n_fraud_needed = int((target_fraud_ratio * n_total - df[fraud_col].sum()) / (1 - target_fraud_ratio))
    n_fraud_needed = max(n_fraud_needed, 1)

    logger.info("Balancing: generating %d extra fraud rows", n_fraud_needed)

    fraud_df = df[df[fraud_col] == 1]
    if len(fraud_df) < 2:
        logger.warning("Too few fraud examples to learn from -- duplicating existing rows")
        extra = fraud_df.sample(n=n_fraud_needed, replace=True).reset_index(drop=True)
    else:
        id_cols = _detect_id_columns(fraud_df)
        work = fraud_df.drop(columns=id_cols) if id_cols else fraud_df.copy()
        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(work)
        synth = GaussianCopulaSynthesizer(metadata)
        synth.fit(work)
        extra = synth.sample(num_rows=n_fraud_needed)
        if id_cols:
            for col in id_cols:
                extra.insert(0, col, [fake.uuid4()[:12].upper() for _ in range(len(extra))])

    balanced = pd.concat([df, extra], ignore_index=True)
    logger.info("Balanced dataset: %d rows, fraud ratio %.2f%%", len(balanced), balanced[fraud_col].mean() * 100)
    return balanced


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


def _detect_id_columns(df: pd.DataFrame) -> list[str]:
    """Heuristic: columns whose cardinality is >90 % of row count are IDs."""
    ids = []
    for col in df.columns:
        if df[col].dtype == object and df[col].nunique() > 0.9 * len(df) and len(df) > 20:
            ids.append(col)
    return ids


def _balance_fraud(
    synthetic_df: pd.DataFrame,
    fraud_col: str,
    synthesizer: GaussianCopulaSynthesizer,
    metadata: SingleTableMetadata,
    id_cols: list[str],
    target_ratio: float = 0.10,
) -> pd.DataFrame:
    """Ensure synthetic dataset has at least *target_ratio* fraud."""
    current = synthetic_df[fraud_col].mean()
    if current >= target_ratio:
        return synthetic_df

    n_extra = int(target_ratio * len(synthetic_df) - synthetic_df[fraud_col].sum())
    if n_extra <= 0:
        return synthetic_df

    logger.info("Topping up %d fraud rows in synthetic dataset", n_extra)

    # Generate extra and force fraud label
    extra = synthesizer.sample(num_rows=n_extra * 3)  # oversample, then filter
    extra[fraud_col] = 1
    extra = extra.head(n_extra)

    if id_cols:
        for col in id_cols:
            if col in extra.columns:
                extra[col] = [fake.uuid4()[:12].upper() for _ in range(len(extra))]

    return pd.concat([synthetic_df, extra], ignore_index=True)


def _compute_statistics(df: pd.DataFrame, fraud_col: str | None) -> dict[str, Any]:
    stats: dict[str, Any] = {
        "row_count": len(df),
        "column_count": len(df.columns),
        "columns": df.columns.tolist(),
    }

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    stats["numeric_summary"] = {}
    for col in numeric_cols:
        stats["numeric_summary"][col] = {
            "mean": round(float(df[col].mean()), 4),
            "std": round(float(df[col].std()), 4),
            "min": round(float(df[col].min()), 4),
            "max": round(float(df[col].max()), 4),
        }

    if fraud_col and fraud_col in df.columns:
        stats["fraud_ratio"] = round(float(df[fraud_col].mean()), 4)
        stats["fraud_count"] = int(df[fraud_col].sum())
        stats["normal_count"] = int(len(df) - df[fraud_col].sum())

    return stats
