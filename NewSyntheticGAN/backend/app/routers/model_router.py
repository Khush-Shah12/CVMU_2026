"""
Model Training Router
=====================
Trains fraud detection ML models and returns evaluation metrics.

Endpoints:
  POST /api/model/train
"""

import json
import logging
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models.dataset_model import Dataset as DatasetORM
from app.models.training_result_model import (
    ModelMetrics,
    TrainRequest,
    TrainResponse,
    TrainingResult,
)
from app.services.fraud_model_service import train_and_evaluate
from app.utils.data_cleaning import clean_dataframe

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/model", tags=["Model Training"])


@router.post("/train", response_model=TrainResponse, status_code=200)
async def train_models(
    body: TrainRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Train Logistic Regression, Random Forest, and XGBoost classifiers on
    the specified dataset and return evaluation metrics for each model.
    """
    # ── Load dataset ────────────────────────────────────────────────────
    result = await db.execute(
        select(DatasetORM).where(DatasetORM.id == body.dataset_id)
    )
    dataset = result.scalar_one_or_none()
    if dataset is None:
        raise HTTPException(
            status_code=404, detail=f"Dataset {body.dataset_id} not found"
        )

    src_path = Path(dataset.file_path)
    if not src_path.exists():
        raise HTTPException(status_code=404, detail="Dataset file missing from disk")

    df = (
        pd.read_csv(src_path) if src_path.suffix.lower() == ".csv"
        else pd.read_excel(src_path)
    )
    df = clean_dataframe(df)

    # ── Validate label column ───────────────────────────────────────────
    if body.label_column not in df.columns:
        raise HTTPException(
            status_code=400,
            detail=f"Label column '{body.label_column}' not found. "
                   f"Available columns: {df.columns.tolist()}",
        )

    # ── Train & evaluate ────────────────────────────────────────────────
    try:
        results = train_and_evaluate(df, label_column=body.label_column)
    except Exception as exc:
        logger.exception("Model training failed")
        raise HTTPException(
            status_code=500, detail=f"Training error: {exc}"
        ) from exc

    # ── Persist results ─────────────────────────────────────────────────
    model_metrics: list[ModelMetrics] = []
    for res in results:
        tr = TrainingResult(
            dataset_id=body.dataset_id,
            model_name=res["model_name"],
            accuracy=res["accuracy"],
            precision_score=res["precision"],
            recall=res["recall"],
            f1=res["f1"],
            confusion_matrix=json.dumps(res["confusion_matrix"]),
        )
        db.add(tr)

        model_metrics.append(
            ModelMetrics(
                model_name=res["model_name"],
                accuracy=res["accuracy"],
                precision=res["precision"],
                recall=res["recall"],
                f1=res["f1"],
                confusion_matrix=res["confusion_matrix"],
            )
        )

    await db.flush()

    return TrainResponse(
        dataset_id=body.dataset_id,
        row_count=len(df),
        models=model_metrics,
    )
