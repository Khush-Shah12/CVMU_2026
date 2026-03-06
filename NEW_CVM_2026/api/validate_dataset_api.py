"""
api/validate_dataset_api.py
===========================
FastAPI router for the synthetic data validation endpoint.

Endpoint:
  POST /validate-data
  Body: multipart/form-data with a CSV file upload
  Response: { realism_score, anomaly_score, fraud_patterns_detected, report }
"""

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel

from services.data_pipeline import run_validation_pipeline

# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

router = APIRouter(tags=["Validation"])


# ---------------------------------------------------------------------------
# Response schema
# ---------------------------------------------------------------------------

class ValidationResponse(BaseModel):
    """Output schema for the /validate-data endpoint."""
    realism_score: float
    anomaly_score: float
    fraud_patterns_detected: bool
    report: dict


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------

@router.post("/validate-data", response_model=ValidationResponse)
async def validate_data(file: UploadFile = File(...)):
    """
    Validate an uploaded CSV dataset of financial transactions.

    Upload a CSV with at least the columns:
      transaction_id, sender_account, receiver_account, amount, timestamp

    Returns a realism score (0-100), anomaly score, fraud detection flag,
    and a detailed quality report.
    """
    # Basic file validation
    if not file.filename.endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail="Only CSV files are supported. Please upload a .csv file.",
        )

    try:
        contents = await file.read()
        if len(contents) == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        result = run_validation_pipeline(
            file_bytes=contents,
            filename=file.filename,
        )
        return result

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Validation failed: {str(e)}")
