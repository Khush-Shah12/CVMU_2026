"""
services/data_pipeline.py
=========================
Glue layer between the API endpoints and the AI modules.

Provides two top-level pipeline functions that the API routers call:
  • run_generation_pipeline   – generate synthetic financial data
  • run_validation_pipeline   – validate an uploaded dataset
"""

import io
from typing import Optional

import pandas as pd

from ai.generator import generate_synthetic_dataset
from ai.validator import validate_dataset


# ============================================================================
# Generation pipeline
# ============================================================================

def run_generation_pipeline(
    dataset_size: int = 10_000,
    fraud_ratio: float = 0.02,
    export_format: Optional[str] = None,
) -> dict:
    """
    Run the full synthetic data generation pipeline.

    Parameters
    ----------
    dataset_size  : number of transactions to generate
    fraud_ratio   : fraction of fraudulent transactions (0.0–0.3)
    export_format : "csv" | "json" | None

    Returns
    -------
    dict containing customers, accounts, transactions, summary,
    and optionally export_paths.
    """
    return generate_synthetic_dataset(
        dataset_size=dataset_size,
        fraud_ratio=fraud_ratio,
        export_format=export_format,
    )


# ============================================================================
# Validation pipeline
# ============================================================================

def run_validation_pipeline(
    file_bytes: bytes,
    filename: str = "upload.csv",
) -> dict:
    """
    Parse an uploaded CSV file and run the full validation suite.

    Parameters
    ----------
    file_bytes : raw bytes of the uploaded CSV
    filename   : original filename (used to infer format)

    Returns
    -------
    dict with realism_score, anomaly_score, fraud_patterns_detected, report
    """
    # Read CSV into DataFrame
    df = pd.read_csv(io.BytesIO(file_bytes))

    if df.empty:
        return {
            "realism_score": 0,
            "anomaly_score": 0,
            "fraud_patterns_detected": False,
            "report": {"summary": "Uploaded dataset is empty.", "statistical": {}, "logical": {}, "fraud_detection": {}},
        }

    return validate_dataset(df)
