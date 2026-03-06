"""
Dataset Router
==============
Handles file uploads, dataset analysis, and dataset comparison.

Endpoints:
  POST /api/dataset/upload
  GET  /api/dataset/{dataset_id}/analysis
  GET  /api/dataset/compare
"""

import logging
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.database import get_db
from app.models.dataset_model import (
    ComparisonReport,
    DatasetAnalysisReport,
    DatasetResponse,
)
from app.models.dataset_model import Dataset as DatasetORM
from app.services.dataset_analysis_service import analyze_dataset
from app.services.dataset_comparison_service import compare_datasets
from app.utils.data_cleaning import clean_dataframe

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/dataset", tags=["Dataset"])


# ── POST /api/dataset/upload ───────────────────────────────────────────
@router.post("/upload", response_model=DatasetResponse, status_code=201)
async def upload_dataset(
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
):
    """
    Upload a CSV or Excel financial transaction dataset.

    The file is validated, stored on disk, and a dataset record is created
    in the database.
    """
    # ── Validate extension ──────────────────────────────────────────────
    filename = file.filename or "upload.csv"
    suffix = Path(filename).suffix.lower()
    if suffix not in (".csv", ".xlsx", ".xls"):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Use CSV or Excel.",
        )

    # ── Save to disk ────────────────────────────────────────────────────
    upload_dir = settings.UPLOAD_DIR
    upload_dir.mkdir(parents=True, exist_ok=True)

    # Avoid filename collisions
    import uuid

    unique_name = f"{uuid.uuid4().hex[:8]}_{filename}"
    file_path = upload_dir / unique_name

    contents = await file.read()
    file_path.write_bytes(contents)
    logger.info("Saved upload → %s (%d bytes)", file_path, len(contents))

    # ── Load & validate ─────────────────────────────────────────────────
    try:
        df = _read_file(file_path)
    except Exception as exc:
        file_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=f"Cannot parse file: {exc}") from exc

    df = clean_dataframe(df)

    # ── Compute fraud ratio ─────────────────────────────────────────────
    fraud_col = _find_fraud_column(df)
    fraud_ratio = round(float(df[fraud_col].mean()), 4) if fraud_col else 0.0

    # ── Persist to DB ───────────────────────────────────────────────────
    dataset = DatasetORM(
        filename=filename,
        file_path=str(file_path),
        row_count=len(df),
        column_count=len(df.columns),
        fraud_ratio=fraud_ratio,
        is_synthetic=False,
    )
    db.add(dataset)
    await db.flush()
    await db.refresh(dataset)

    logger.info("Dataset %d created -- %d rows", dataset.id, dataset.row_count)
    return dataset


# ── GET /api/dataset/{dataset_id}/analysis ─────────────────────────────
@router.get("/{dataset_id}/analysis", response_model=DatasetAnalysisReport)
async def get_dataset_analysis(
    dataset_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Run a full statistical analysis on the specified dataset."""
    dataset = await _get_dataset_or_404(dataset_id, db)
    df = _read_file(Path(dataset.file_path))
    report = analyze_dataset(df, dataset_id)
    return report


# ── GET /api/dataset/compare ──────────────────────────────────────────
@router.get("/compare", response_model=ComparisonReport)
async def compare_two_datasets(
    original_id: int = Query(..., description="ID of the original dataset"),
    synthetic_id: int = Query(..., description="ID of the synthetic dataset"),
    db: AsyncSession = Depends(get_db),
):
    """Compare an original dataset against a synthetic one."""
    original = await _get_dataset_or_404(original_id, db)
    synthetic = await _get_dataset_or_404(synthetic_id, db)

    orig_df = _read_file(Path(original.file_path))
    syn_df = _read_file(Path(synthetic.file_path))

    report = compare_datasets(orig_df, syn_df, original_id, synthetic_id)
    return report


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

_FRAUD_ALIASES = {"fraudlabel", "fraud", "is_fraud", "isfraud", "label", "fraud_flag"}


def _find_fraud_column(df: pd.DataFrame) -> str | None:
    for col in df.columns:
        if col.lower().replace(" ", "").replace("_", "") in {
            a.replace("_", "") for a in _FRAUD_ALIASES
        }:
            return col
    return None


def _read_file(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    return pd.read_excel(path)


async def _get_dataset_or_404(
    dataset_id: int, db: AsyncSession
) -> DatasetORM:
    result = await db.execute(select(DatasetORM).where(DatasetORM.id == dataset_id))
    dataset = result.scalar_one_or_none()
    if dataset is None:
        raise HTTPException(status_code=404, detail=f"Dataset {dataset_id} not found")
    return dataset
