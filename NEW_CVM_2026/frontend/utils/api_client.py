"""
frontend/utils/api_client.py
============================
Centralised HTTP client for communicating with the FastAPI backend.
All API calls go through this module so that error handling,
base URL, and timeouts are managed in one place.
"""

import requests
import streamlit as st

BASE_URL = "http://localhost:8000"
TIMEOUT = 300  # seconds (generation can be slow)


def _handle_response(response: requests.Response) -> dict | bytes:
    """Check response status and return JSON or raise a friendly error."""
    if response.status_code == 200:
        content_type = response.headers.get("content-type", "")
        if "text/csv" in content_type or "octet-stream" in content_type:
            return response.content  # raw bytes for file downloads
        return response.json()
    else:
        try:
            detail = response.json().get("detail", response.text)
        except Exception:
            detail = response.text
        raise Exception(f"API Error ({response.status_code}): {detail}")


# --------------------------------------------------------------------------
# 1. Upload dataset
# --------------------------------------------------------------------------

def upload_dataset(file_bytes: bytes, filename: str) -> dict:
    """POST /upload-dataset — upload a CSV file."""
    resp = requests.post(
        f"{BASE_URL}/upload-dataset",
        files={"file": (filename, file_bytes, "text/csv")},
        timeout=TIMEOUT,
    )
    return _handle_response(resp)


# --------------------------------------------------------------------------
# 2. Generate synthetic data
# --------------------------------------------------------------------------

def generate_data(dataset_id: str, num_rows: int | None = None, fraud_ratio: float = 0.05) -> dict:
    """POST /generate-data — generate synthetic data from uploaded dataset."""
    payload = {"dataset_id": dataset_id, "fraud_ratio": fraud_ratio}
    if num_rows is not None:
        payload["num_rows"] = num_rows
    resp = requests.post(
        f"{BASE_URL}/generate-data",
        json=payload,
        timeout=TIMEOUT,
    )
    return _handle_response(resp)


# --------------------------------------------------------------------------
# 3. Validate synthetic data
# --------------------------------------------------------------------------

def validate_data(dataset_id: str) -> dict:
    """POST /validate-data — validate synthetic vs original dataset."""
    resp = requests.post(
        f"{BASE_URL}/validate-data",
        json={"dataset_id": dataset_id},
        timeout=TIMEOUT,
    )
    return _handle_response(resp)


# --------------------------------------------------------------------------
# 4. Dataset stats / analytics
# --------------------------------------------------------------------------

def get_dataset_stats(dataset_id: str) -> dict:
    """GET /dataset-stats/{dataset_id} — get dataset analytics."""
    resp = requests.get(
        f"{BASE_URL}/dataset-stats/{dataset_id}",
        timeout=TIMEOUT,
    )
    return _handle_response(resp)


# --------------------------------------------------------------------------
# 5. Download synthetic dataset
# --------------------------------------------------------------------------

def download_synthetic(dataset_id: str) -> bytes:
    """GET /download-synthetic/{dataset_id} — download generated CSV."""
    resp = requests.get(
        f"{BASE_URL}/download-synthetic/{dataset_id}",
        timeout=TIMEOUT,
    )
    result = _handle_response(resp)
    if isinstance(result, bytes):
        return result
    raise Exception("Expected file download but got JSON response.")
