"""
routes/analytics.py
===================
GET /dataset-stats/{dataset_id}

Return detailed statistics about a dataset:
  - total transactions, columns
  - fraud vs normal ratio
  - missing values
  - summary statistics for numeric columns
"""

import logging

import numpy as np
from fastapi import APIRouter, HTTPException

from models_schema.dataset_model import DatasetStatsResponse
from utils.file_handler import dataset_exists, load_dataset
import helpers

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Analytics"])


@router.get("/dataset-stats/{dataset_id}", response_model=DatasetStatsResponse)
async def dataset_stats(dataset_id: str):
    """
    Get detailed statistics about an uploaded or generated dataset.

    If both original and synthetic exist, stats are for the original.
    Append ?type=synthetic to get synthetic stats.
    """
    # Determine which dataset to analyze
    has_original = dataset_exists(dataset_id, kind="original")
    has_synthetic = dataset_exists(dataset_id, kind="synthetic")

    if not has_original and not has_synthetic:
        raise HTTPException(
            status_code=404,
            detail=f"Dataset '{dataset_id}' not found.",
        )

    # Default to original if available
    kind = "original" if has_original else "synthetic"
    dataset_type = "both" if (has_original and has_synthetic) else kind

    try:
        df = load_dataset(dataset_id, kind=kind)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' ({kind}) not found.")

    # Basic counts
    total_transactions = len(df)
    columns = len(df.columns)
    column_names = list(df.columns)
    missing_values = int(df.isnull().sum().sum())

    # Fraud ratio
    fraud_ratio = None
    normal_count = None
    fraud_count = None
    if "is_fraud" in df.columns:
        fraud_count = int(df["is_fraud"].sum())
        normal_count = total_transactions - fraud_count
        fraud_ratio = round(fraud_count / max(total_transactions, 1), 4)

    # Summary statistics for numeric columns
    numeric_df = df.select_dtypes(include=[np.number])
    summary_dict = {}
    for col in numeric_df.columns:
        col_data = numeric_df[col].dropna()
        summary_dict[col] = {
            "count": int(col_data.count()),
            "mean": round(float(col_data.mean()), 2),
            "std": round(float(col_data.std()), 2),
            "min": round(float(col_data.min()), 2),
            "max": round(float(col_data.max()), 2),
            "median": round(float(col_data.median()), 2),
        }

    summary_dict = helpers.sanitise_for_json(summary_dict)

    return DatasetStatsResponse(
        dataset_id=dataset_id,
        dataset_type=dataset_type,
        total_transactions=total_transactions,
        columns=columns,
        column_names=column_names,
        fraud_ratio=fraud_ratio,
        normal_count=normal_count,
        fraud_count=fraud_count,
        missing_values=missing_values,
        summary=summary_dict,
    )
