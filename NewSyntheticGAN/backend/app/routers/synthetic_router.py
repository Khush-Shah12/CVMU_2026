"""
Synthetic Data Router
=====================
Handles synthetic dataset generation via SDV.

Endpoints:
  POST /api/synthetic/generate
"""

import logging
import uuid
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.database import get_db
from app.models.dataset_model import (
    Dataset as DatasetORM,
    GenerateSyntheticRequest,
    GenerateSyntheticResponse,
)
from app.services.synthetic_generation_service import generate_synthetic_data
from app.utils.data_cleaning import clean_dataframe

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/synthetic", tags=["Synthetic Data"])


@router.post("/generate", response_model=GenerateSyntheticResponse, status_code=201)
async def generate_synthetic(
    body: GenerateSyntheticRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Generate synthetic financial transactions based on an existing dataset.

    The synthetic data preserves statistical distributions and correlations
    from the source dataset while removing sensitive identifiers.
    """
    # ── Load source dataset ─────────────────────────────────────────────
    result = await db.execute(
        select(DatasetORM).where(DatasetORM.id == body.dataset_id)
    )
    source = result.scalar_one_or_none()
    if source is None:
        raise HTTPException(
            status_code=404, detail=f"Source dataset {body.dataset_id} not found"
        )

    src_path = Path(source.file_path)
    if not src_path.exists():
        raise HTTPException(status_code=404, detail="Source file missing from disk")

    df = (
        pd.read_csv(src_path) if src_path.suffix.lower() == ".csv"
        else pd.read_excel(src_path)
    )
    df = clean_dataframe(df)

    # ── Generate ────────────────────────────────────────────────────────
    try:
        synthetic_df, stats = generate_synthetic_data(df, num_samples=body.num_samples)
    except Exception as exc:
        logger.exception("Synthetic generation failed")
        raise HTTPException(
            status_code=500, detail=f"Generation error: {exc}"
        ) from exc

    # ── Save synthetic dataset ──────────────────────────────────────────
    gen_dir = settings.GENERATED_DIR
    gen_dir.mkdir(parents=True, exist_ok=True)

    syn_filename = f"synthetic_{uuid.uuid4().hex[:8]}.csv"
    syn_path = gen_dir / syn_filename
    synthetic_df.to_csv(syn_path, index=False)
    logger.info("Saved synthetic dataset → %s (%d rows)", syn_path, len(synthetic_df))

    # ── Fraud ratio ─────────────────────────────────────────────────────
    fraud_col = _find_fraud_column(synthetic_df)
    fraud_ratio = round(float(synthetic_df[fraud_col].mean()), 4) if fraud_col else 0.0

    # ── DB record ───────────────────────────────────────────────────────
    dataset = DatasetORM(
        filename=syn_filename,
        file_path=str(syn_path),
        row_count=len(synthetic_df),
        column_count=len(synthetic_df.columns),
        fraud_ratio=fraud_ratio,
        is_synthetic=True,
        parent_dataset_id=body.dataset_id,
    )
    db.add(dataset)
    await db.flush()
    await db.refresh(dataset)

    return GenerateSyntheticResponse(
        synthetic_dataset_id=dataset.id,
        num_samples_generated=len(synthetic_df),
        fraud_ratio=fraud_ratio,
        dataset_statistics=stats,
    )


# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════

_FRAUD_ALIASES = {"fraudlabel", "fraud", "is_fraud", "isfraud", "label", "fraud_flag"}


def _find_fraud_column(df: pd.DataFrame) -> str | None:
    for col in df.columns:
        if col.lower().replace(" ", "").replace("_", "") in {
            a.replace("_", "") for a in _FRAUD_ALIASES
        }:
            return col
    return None
