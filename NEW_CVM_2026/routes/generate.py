"""
routes/generate.py
==================
POST /generate-data

Accept a dataset_id and generation parameters, run the AI generator,
save the synthetic dataset, and return the result status.
"""

import logging

from fastapi import APIRouter, HTTPException

from models_schema.dataset_model import GenerateRequest, GenerateResponse
from services.generator_service import generate_synthetic_data
from utils.file_handler import dataset_exists

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Generation"])


@router.post("/generate-data", response_model=GenerateResponse)
async def generate_data(request: GenerateRequest):
    """
    Generate synthetic financial data from an uploaded original dataset.

    Supply the dataset_id from a previous upload, optionally specify
    the number of synthetic rows and fraud ratio.
    """
    # Check that the original dataset exists
    if not dataset_exists(request.dataset_id, kind="original"):
        raise HTTPException(
            status_code=404,
            detail=f"Dataset '{request.dataset_id}' not found. Please upload it first via /upload-dataset.",
        )

    try:
        result = generate_synthetic_data(
            dataset_id=request.dataset_id,
            num_rows=request.num_rows,
            fraud_ratio=request.fraud_ratio,
        )
        return GenerateResponse(
            dataset_id=result["dataset_id"],
            synthetic_rows=result["synthetic_rows"],
            fraud_rows=result["fraud_rows"],
            status=result["status"],
        )
    except Exception as e:
        logger.exception(f"Generation failed for dataset '{request.dataset_id}'")
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")
