"""
routes/download.py
==================
GET /download-synthetic/{dataset_id}

Serve the generated synthetic dataset as a downloadable CSV file.
"""

import logging

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from utils.file_handler import dataset_exists, get_synthetic_path

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Download"])


@router.get("/download-synthetic/{dataset_id}")
async def download_synthetic(dataset_id: str):
    """
    Download the generated synthetic dataset as a CSV file.

    The dataset must have been previously generated via /generate-data.
    """
    if not dataset_exists(dataset_id, kind="synthetic"):
        raise HTTPException(
            status_code=404,
            detail=f"Synthetic dataset for '{dataset_id}' not found. Please generate it first via /generate-data.",
        )

    path = get_synthetic_path(dataset_id)
    filename = f"{dataset_id}_synthetic.csv"

    logger.info(f"Serving synthetic dataset download: {filename}")

    return FileResponse(
        path=path,
        media_type="text/csv",
        filename=filename,
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )
