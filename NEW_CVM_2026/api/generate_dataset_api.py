"""
api/generate_dataset_api.py
===========================
FastAPI router for the synthetic data generation endpoint.

Endpoint:
  POST /generate-data
  Body: { "dataset_size": int, "fraud_ratio": float, "export_format": str|null }
  Response: generated customers, accounts, transactions + summary
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from services.data_pipeline import run_generation_pipeline
import config

# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

router = APIRouter(tags=["Generation"])


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class GenerateRequest(BaseModel):
    """Input schema for the /generate-data endpoint."""
    dataset_size: int = Field(
        default=config.DEFAULT_DATASET_SIZE,
        ge=config.MIN_DATASET_SIZE,
        le=config.MAX_DATASET_SIZE,
        description="Total number of transactions to generate",
    )
    fraud_ratio: float = Field(
        default=config.DEFAULT_FRAUD_RATIO,
        ge=config.MIN_FRAUD_RATIO,
        le=config.MAX_FRAUD_RATIO,
        description="Fraction of fraudulent transactions (0.0–0.3)",
    )
    export_format: str | None = Field(
        default=None,
        description="Optional export format: 'csv' or 'json'",
    )


class GenerateSummary(BaseModel):
    """Summary statistics returned alongside the dataset."""
    total_customers: int
    total_accounts: int
    total_transactions: int
    fraud_transactions: int
    fraud_ratio_actual: float
    dataset_size_requested: int


class GenerateResponse(BaseModel):
    """Full response schema for /generate-data."""
    customers: list[dict]
    accounts: list[dict]
    transactions: list[dict]
    summary: GenerateSummary
    export_paths: dict | None = None


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------

@router.post("/generate-data", response_model=GenerateResponse)
async def generate_data(request: GenerateRequest):
    """
    Generate a synthetic financial dataset.

    Supply `dataset_size` (number of transactions) and `fraud_ratio` (0.0–0.3).
    Optionally set `export_format` to "csv" or "json" to also write files to disk.
    """
    try:
        result = run_generation_pipeline(
            dataset_size=request.dataset_size,
            fraud_ratio=request.fraud_ratio,
            export_format=request.export_format,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")
