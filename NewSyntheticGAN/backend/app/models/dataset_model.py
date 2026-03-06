"""
SQLAlchemy ORM models and Pydantic schemas for datasets.
"""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field
from sqlalchemy import Boolean, Column, DateTime, Float, Integer, String, Text, func

from app.database import Base


# ═══════════════════════════════════════════════════════════════════════════
# SQLAlchemy ORM Model
# ═══════════════════════════════════════════════════════════════════════════

class Dataset(Base):
    """Represents an uploaded or generated dataset."""

    __tablename__ = "datasets"

    id = Column(Integer, primary_key=True, autoincrement=True)
    filename = Column(String(255), nullable=False)
    file_path = Column(Text, nullable=False)
    row_count = Column(Integer, default=0)
    column_count = Column(Integer, default=0)
    fraud_ratio = Column(Float, default=0.0)
    is_synthetic = Column(Boolean, default=False)
    parent_dataset_id = Column(Integer, nullable=True)
    created_at = Column(DateTime, server_default=func.now())


# ═══════════════════════════════════════════════════════════════════════════
# Pydantic Response Schemas
# ═══════════════════════════════════════════════════════════════════════════

class DatasetResponse(BaseModel):
    """Returned after upload or generation."""

    id: int
    filename: str
    row_count: int
    column_count: int
    fraud_ratio: float
    is_synthetic: bool
    parent_dataset_id: Optional[int] = None
    created_at: datetime

    model_config = {"from_attributes": True}


class AmountDistribution(BaseModel):
    mean: float
    median: float
    std: float
    min: float
    max: float
    q25: float
    q75: float


class DatasetAnalysisReport(BaseModel):
    """Full analysis report for a dataset."""

    dataset_id: int
    row_count: int
    column_count: int
    columns: list[str]
    missing_values: dict[str, int]
    duplicate_rows: int
    fraud_ratio: float
    fraud_count: int
    normal_count: int
    amount_distribution: AmountDistribution
    outlier_count: int
    outlier_transactions: list[dict]
    suspicious_patterns: list[str]


class GenerateSyntheticRequest(BaseModel):
    """Request body for synthetic generation."""

    dataset_id: int = Field(..., description="ID of the source dataset")
    num_samples: int = Field(
        default=1000, ge=10, le=100000,
        description="Number of synthetic rows to generate",
    )


class GenerateSyntheticResponse(BaseModel):
    """Response after synthetic generation."""

    synthetic_dataset_id: int
    num_samples_generated: int
    fraud_ratio: float
    dataset_statistics: dict


class ComparisonReport(BaseModel):
    """Statistical comparison between original and synthetic datasets."""

    original_dataset_id: int
    synthetic_dataset_id: int
    column_distribution_similarity: dict[str, float]
    overall_distribution_score: float
    correlation_similarity: float
    fraud_ratio_original: float
    fraud_ratio_synthetic: float
    fraud_distribution_preserved: bool
