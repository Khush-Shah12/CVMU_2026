"""
models_schema/dataset_model.py
==============================
Pydantic request/response schemas for all API endpoints.
Centralised here so routes stay clean and schemas are reusable.
"""

from pydantic import BaseModel, Field
from typing import Optional


# ============================================================================
# Upload
# ============================================================================

class UploadResponse(BaseModel):
    """Response from POST /upload-dataset."""
    dataset_id: str
    rows: int
    columns: int
    column_names: list[str]
    message: str


# ============================================================================
# Generate
# ============================================================================

class GenerateRequest(BaseModel):
    """Request body for POST /generate-data."""
    dataset_id: str = Field(..., description="ID of the uploaded original dataset")
    num_rows: Optional[int] = Field(
        default=None,
        ge=100,
        le=5_000_000,
        description="Number of synthetic rows to generate. Defaults to same size as original.",
    )
    fraud_ratio: float = Field(
        default=0.02,
        ge=0.0,
        le=0.30,
        description="Fraction of fraudulent transactions (0.0 - 0.3)",
    )


class GenerateResponse(BaseModel):
    """Response from POST /generate-data."""
    dataset_id: str
    synthetic_rows: int
    fraud_rows: int
    status: str


# ============================================================================
# Validate
# ============================================================================

class ValidateRequest(BaseModel):
    """Request body for POST /validate-data."""
    dataset_id: str = Field(..., description="ID of the dataset to validate")


class ValidateResponse(BaseModel):
    """Response from POST /validate-data."""
    dataset_id: str
    similarity_score: float
    fraud_ratio_match: str
    correlation_match: str
    quality_score: float
    realism_score: float
    anomaly_score: float
    fraud_patterns_detected: bool
    detailed_report: dict


# ============================================================================
# Download — no request/response model needed (returns file)
# ============================================================================


# ============================================================================
# Dataset Stats / Analytics
# ============================================================================

class DatasetStatsResponse(BaseModel):
    """Response from GET /dataset-stats/{dataset_id}."""
    dataset_id: str
    dataset_type: str  # "original" | "synthetic" | "both"
    total_transactions: int
    columns: int
    column_names: list[str]
    fraud_ratio: Optional[float] = None
    normal_count: Optional[int] = None
    fraud_count: Optional[int] = None
    missing_values: int
    summary: dict
