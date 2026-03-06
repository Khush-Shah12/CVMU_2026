"""
routes/upload.py
================
POST /upload-dataset

Accept a CSV file from the frontend, validate it, store it in
datasets/original/, and return a dataset ID with basic statistics.
"""

import logging

from fastapi import APIRouter, File, HTTPException, UploadFile

from models_schema.dataset_model import UploadResponse
from utils.file_handler import (
    generate_dataset_id,
    validate_csv_file,
    save_uploaded_file,
    load_dataset,
    get_dataset_basic_stats,
)

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Upload"])


@router.post("/upload-dataset", response_model=UploadResponse)
async def upload_dataset(file: UploadFile = File(...)):
    """
    Upload a CSV dataset.

    The file is validated for format and size, then stored with a unique
    dataset ID. Returns the ID and basic statistics about the dataset.
    """
    content = await file.read()
    filename = file.filename or "unknown.csv"

    # Validate
    is_valid, msg = validate_csv_file(content, filename)
    if not is_valid:
        raise HTTPException(status_code=400, detail=msg)

    # Generate unique ID and save
    dataset_id = generate_dataset_id()
    save_uploaded_file(content, dataset_id)

    # Load and compute stats
    df = load_dataset(dataset_id, kind="original")
    stats = get_dataset_basic_stats(df)

    logger.info(f"Dataset uploaded: {dataset_id} ({stats['rows']} rows, {stats['columns']} cols)")

    return UploadResponse(
        dataset_id=dataset_id,
        rows=stats["rows"],
        columns=stats["columns"],
        column_names=stats["column_names"],
        message="Dataset uploaded successfully",
    )
