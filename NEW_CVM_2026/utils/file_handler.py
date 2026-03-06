"""
utils/file_handler.py
=====================
File-handling utilities for the backend:
  - Unique dataset ID generation
  - CSV validation (format, size, encoding)
  - Path helpers for original / synthetic datasets
  - Safe file read / write operations
"""

import os
import uuid
import logging
from datetime import datetime

import pandas as pd

import config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dataset directory management
# ---------------------------------------------------------------------------

def ensure_dataset_dirs() -> None:
    """Create the datasets/original/ and datasets/synthetic/ directories."""
    os.makedirs(config.ORIGINAL_DATASET_DIR, exist_ok=True)
    os.makedirs(config.SYNTHETIC_DATASET_DIR, exist_ok=True)


def get_original_path(dataset_id: str) -> str:
    """Return the full path for an original dataset CSV."""
    return os.path.join(config.ORIGINAL_DATASET_DIR, f"{dataset_id}.csv")


def get_synthetic_path(dataset_id: str) -> str:
    """Return the full path for a synthetic dataset CSV."""
    return os.path.join(config.SYNTHETIC_DATASET_DIR, f"{dataset_id}_synthetic.csv")


def dataset_exists(dataset_id: str, kind: str = "original") -> bool:
    """Check whether a dataset file exists. kind = 'original' | 'synthetic'."""
    if kind == "synthetic":
        return os.path.isfile(get_synthetic_path(dataset_id))
    return os.path.isfile(get_original_path(dataset_id))


# ---------------------------------------------------------------------------
# Unique ID generation
# ---------------------------------------------------------------------------

def generate_dataset_id() -> str:
    """
    Generate a short, unique, human-readable dataset ID.
    Format: ds_<8-hex-chars>
    """
    return f"ds_{uuid.uuid4().hex[:8]}"


# ---------------------------------------------------------------------------
# CSV validation
# ---------------------------------------------------------------------------

def validate_csv_file(content: bytes, filename: str) -> tuple[bool, str]:
    """
    Validate an uploaded file:
      1. Filename ends with .csv
      2. File size within limit
      3. File is valid UTF-8 text
      4. File is parseable as CSV with at least 1 row and 1 column

    Returns
    -------
    (is_valid, message)
    """
    # Check extension
    if not filename.lower().endswith(".csv"):
        return False, "Only CSV files are accepted. Please upload a .csv file."

    # Check file size
    size_mb = len(content) / (1024 * 1024)
    if size_mb > config.MAX_FILE_SIZE_MB:
        return False, f"File too large ({size_mb:.1f} MB). Maximum allowed is {config.MAX_FILE_SIZE_MB} MB."

    if len(content) == 0:
        return False, "Uploaded file is empty."

    # Check encoding
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError:
        try:
            text = content.decode("latin-1")
        except Exception:
            return False, "File encoding not supported. Please upload a UTF-8 or Latin-1 CSV."

    # Check parseable as CSV
    try:
        import io
        df = pd.read_csv(io.BytesIO(content))
        if df.empty:
            return False, "CSV file is empty (no data rows)."
        if len(df.columns) < 1:
            return False, "CSV file has no columns."
    except Exception as e:
        return False, f"Could not parse CSV: {str(e)}"

    return True, "File is valid."


def save_uploaded_file(content: bytes, dataset_id: str) -> str:
    """
    Save uploaded CSV bytes to datasets/original/<dataset_id>.csv.
    Returns the file path.
    """
    ensure_dataset_dirs()
    path = get_original_path(dataset_id)
    with open(path, "wb") as f:
        f.write(content)
    logger.info(f"Saved uploaded dataset to {path}")
    return path


def load_dataset(dataset_id: str, kind: str = "original") -> pd.DataFrame:
    """
    Load a dataset CSV into a DataFrame.
    kind = 'original' | 'synthetic'
    """
    if kind == "synthetic":
        path = get_synthetic_path(dataset_id)
    else:
        path = get_original_path(dataset_id)

    if not os.path.isfile(path):
        raise FileNotFoundError(f"Dataset '{dataset_id}' ({kind}) not found at {path}")

    df = pd.read_csv(path)
    logger.info(f"Loaded {kind} dataset '{dataset_id}': {len(df)} rows, {len(df.columns)} columns")
    return df


def save_synthetic_dataset(df: pd.DataFrame, dataset_id: str) -> str:
    """
    Save a synthetic DataFrame to datasets/synthetic/<dataset_id>_synthetic.csv.
    Returns the file path.
    """
    ensure_dataset_dirs()
    path = get_synthetic_path(dataset_id)
    df.to_csv(path, index=False)
    logger.info(f"Saved synthetic dataset to {path}: {len(df)} rows")
    return path


def get_dataset_basic_stats(df: pd.DataFrame) -> dict:
    """Return basic statistics about a DataFrame."""
    return {
        "rows": len(df),
        "columns": len(df.columns),
        "column_names": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "missing_values": int(df.isnull().sum().sum()),
        "missing_per_column": {col: int(v) for col, v in df.isnull().sum().items()},
    }
