"""
SQLAlchemy ORM models and Pydantic schemas for ML training results.
"""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field
from sqlalchemy import Column, DateTime, Float, Integer, String, Text, func

from app.database import Base


# ═══════════════════════════════════════════════════════════════════════════
# SQLAlchemy ORM Model
# ═══════════════════════════════════════════════════════════════════════════

class TrainingResult(Base):
    """Stores evaluation metrics from a trained model."""

    __tablename__ = "training_results"

    id = Column(Integer, primary_key=True, autoincrement=True)
    dataset_id = Column(Integer, nullable=False)
    model_name = Column(String(100), nullable=False)
    accuracy = Column(Float, default=0.0)
    precision_score = Column(Float, default=0.0)
    recall = Column(Float, default=0.0)
    f1 = Column(Float, default=0.0)
    confusion_matrix = Column(Text, nullable=True)  # JSON string
    created_at = Column(DateTime, server_default=func.now())


# ═══════════════════════════════════════════════════════════════════════════
# Pydantic Schemas
# ═══════════════════════════════════════════════════════════════════════════

class TrainRequest(BaseModel):
    """Request body for model training."""

    dataset_id: int = Field(..., description="ID of the dataset to train on")
    label_column: str = Field(
        default="FraudLabel",
        description="Name of the binary label column",
    )


class ModelMetrics(BaseModel):
    """Evaluation metrics for a single model."""

    model_name: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    confusion_matrix: list[list[int]]


class TrainResponse(BaseModel):
    """Response after training all models."""

    dataset_id: int
    row_count: int
    models: list[ModelMetrics]
