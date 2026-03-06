"""
Synthesizer Router
==================
Exposes a two-step workflow around the in-memory GaussianCopulaSynthesizer:

1. Train on a stored dataset (by ID), with optional fraud-class balancing.
2. Generate synthetic rows from the last trained model, optionally saving to CSV.
"""

import logging
import uuid
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.database import get_db
from app.models.dataset_model import Dataset as DatasetORM
from app.services.synthetic_generator import (
    generate_synthetic_data,
    train_synthesizer,
)
from app.utils.data_cleaning import clean_dataframe

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/synthesizer", tags=["Synthetic Generator"])


class SynthTrainRequest(BaseModel):
    """Request body for training the global synthesizer."""

    dataset_id: int = Field(..., description="ID of the dataset to use for training")
    target_column: Optional[str] = Field(
        default=None,
        description=(
            "Name of the binary fraud/label column. "
            "Required if balance is True and auto-detection is not desired."
        ),
    )
    balance: bool = Field(
        default=False,
        description="Whether to upsample minority class before training.",
    )


class SynthTrainResponse(BaseModel):
    """Metadata about the last training run."""

    run_id: str
    rows_trained: int
    columns: list[str]
    balanced: bool
    target_column: Optional[str]
    elapsed_seconds: float
    field_types: dict[str, Any]


class SynthGenerateRequest(BaseModel):
    """Request body for generating synthetic rows."""

    num_rows: int = Field(
        ...,
        ge=1,
        le=100_000,
        description="Number of synthetic rows to generate.",
    )
    save_to_disk: bool = Field(
        default=True,
        description="If true, persist generated rows to CSV under GENERATED_DIR.",
    )


class SynthGenerateResponse(BaseModel):
    """Metadata returned after generation."""

    run_id: str
    generated_rows: int
    saved_to: Optional[str]
    elapsed_seconds: float
    columns: list[str]
    training_meta: dict[str, Any]


async def _get_dataset_or_404(
    dataset_id: int,
    db: AsyncSession,
) -> DatasetORM:
    result = await db.execute(select(DatasetORM).where(DatasetORM.id == dataset_id))
    dataset = result.scalar_one_or_none()
    if dataset is None:
        raise HTTPException(status_code=404, detail=f"Dataset {dataset_id} not found")
    return dataset


def _read_file(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    return pd.read_excel(path)


@router.post("/train", response_model=SynthTrainResponse, status_code=200)
async def train_global_synthesizer(
    body: SynthTrainRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Train the in-memory GaussianCopulaSynthesizer on a stored dataset.

    This endpoint prepares the global synthesizer that can later be used
    by `/api/synthesizer/generate` to sample additional rows.
    """
    dataset = await _get_dataset_or_404(body.dataset_id, db)

    src_path = Path(dataset.file_path)
    if not src_path.exists():
        raise HTTPException(status_code=404, detail="Dataset file missing from disk")

    df = _read_file(src_path)
    df = clean_dataframe(df)

    try:
        meta = train_synthesizer(
            df,
            target_column=body.target_column,
            balance=body.balance,
        )
    except ValueError as exc:
        # Validation issues (e.g. missing target column) are returned as 400s
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("Synthesizer training failed")
        raise HTTPException(
            status_code=500,
            detail=f"Training error: {exc}",
        ) from exc

    return SynthTrainResponse(**meta)


@router.post("/generate", response_model=SynthGenerateResponse, status_code=200)
async def generate_from_global_synthesizer(
    body: SynthGenerateRequest,
):
    """
    Generate synthetic rows from the last trained synthesizer.

    If `save_to_disk` is true, the generated data is saved as a CSV under
    `settings.GENERATED_DIR` and the absolute path is returned in `saved_to`.
    """
    output_path: Optional[Path] = None
    if body.save_to_disk:
        settings.GENERATED_DIR.mkdir(parents=True, exist_ok=True)
        filename = f"global_synth_{uuid.uuid4().hex[:8]}.csv"
        output_path = settings.GENERATED_DIR / filename

    try:
        _, meta = generate_synthetic_data(
            num_rows=body.num_rows,
            output_path=str(output_path) if output_path else None,
        )
    except RuntimeError as exc:
        # Not trained yet.
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("Synthetic generation failed")
        raise HTTPException(
            status_code=500,
            detail=f"Generation error: {exc}",
        ) from exc

    return SynthGenerateResponse(**meta)

