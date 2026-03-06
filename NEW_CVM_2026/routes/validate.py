"""
routes/validate.py
==================
POST /validate-data

Compare an original dataset with its synthetic counterpart.
Returns similarity metrics, fraud ratio comparison, and quality scores.
"""

import logging

from fastapi import APIRouter, HTTPException

from models_schema.dataset_model import ValidateRequest, ValidateResponse
from services.validator_service import validate_synthetic_data
from utils.file_handler import dataset_exists

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Validation"])


@router.post("/validate-data", response_model=ValidateResponse)
async def validate_data(request: ValidateRequest):
    """
    Validate synthetic data quality by comparing it against the original dataset.

    Both original and synthetic datasets must exist for the given dataset_id.
    Returns similarity score, fraud ratio match, correlation match, and quality score.
    """
    # Check that both datasets exist
    if not dataset_exists(request.dataset_id, kind="original"):
        raise HTTPException(
            status_code=404,
            detail=f"Original dataset '{request.dataset_id}' not found.",
        )
    if not dataset_exists(request.dataset_id, kind="synthetic"):
        raise HTTPException(
            status_code=404,
            detail=f"Synthetic dataset for '{request.dataset_id}' not found. Please generate it first via /generate-data.",
        )

    try:
        result = validate_synthetic_data(dataset_id=request.dataset_id)
        return ValidateResponse(**result)
    except Exception as e:
        logger.exception(f"Validation failed for dataset '{request.dataset_id}'")
        raise HTTPException(status_code=500, detail=f"Validation failed: {str(e)}")
