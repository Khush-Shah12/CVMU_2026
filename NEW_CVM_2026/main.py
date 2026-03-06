"""
main.py
=======
FastAPI application entry-point for the
Finance Sector Synthetic Data Generator & Validator Platform.

Production-ready backend with 5 API endpoints:
  POST /upload-dataset       - Upload a CSV dataset
  POST /generate-data        - Generate synthetic data from an uploaded dataset
  POST /validate-data        - Validate synthetic vs original dataset
  GET  /download-synthetic/  - Download generated synthetic CSV
  GET  /dataset-stats/       - Get dataset analytics

Run with:
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload

Interactive docs:
    http://localhost:8000/docs   (Swagger UI)
    http://localhost:8000/redoc  (ReDoc)
"""

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import config
from utils.file_handler import ensure_dataset_dirs

# Import route modules
from routes.upload import router as upload_router
from routes.generate import router as generate_router
from routes.validate import router as validate_router
from routes.download import router as download_router
from routes.analytics import router as analytics_router

# --------------------------------------------------------------------------
# Logging setup
# --------------------------------------------------------------------------

logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL, logging.INFO),
    format=config.LOG_FORMAT,
)
logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# App initialisation
# --------------------------------------------------------------------------

app = FastAPI(
    title="Finance Synthetic Data Generator & Validator",
    description=(
        "Production-ready backend API for generating and validating "
        "synthetic financial transaction datasets.\n\n"
        "**Module 1 - Generator AI**: Creates realistic synthetic financial data "
        "(customers, accounts, transactions) using a Variational Autoencoder.\n\n"
        "**Module 2 - Validator AI**: Evaluates datasets for statistical realism, "
        "logical consistency, and fraud patterns using Isolation Forest & "
        "Autoencoder anomaly detection.\n\n"
        "**Endpoints**:\n"
        "- `POST /upload-dataset` - Upload a CSV dataset\n"
        "- `POST /generate-data` - Generate synthetic data\n"
        "- `POST /validate-data` - Validate synthetic data quality\n"
        "- `GET /download-synthetic/{id}` - Download synthetic CSV\n"
        "- `GET /dataset-stats/{id}` - Get dataset analytics\n"
    ),
    version="2.0.0",
)

# --------------------------------------------------------------------------
# CORS - allow all origins during development
# --------------------------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------------------------------
# Create dataset directories on startup
# --------------------------------------------------------------------------

@app.on_event("startup")
async def startup_event():
    ensure_dataset_dirs()
    logger.info("Dataset directories ready.")
    logger.info("Finance Synthetic Data Platform started successfully.")

# --------------------------------------------------------------------------
# Mount API routers
# --------------------------------------------------------------------------

app.include_router(upload_router)
app.include_router(generate_router)
app.include_router(validate_router)
app.include_router(download_router)
app.include_router(analytics_router)

# --------------------------------------------------------------------------
# Health check
# --------------------------------------------------------------------------

@app.get("/", tags=["Health"])
async def root():
    """Health-check / welcome endpoint."""
    return {
        "status": "running",
        "platform": "Finance Synthetic Data Generator & Validator",
        "version": "2.0.0",
        "endpoints": {
            "upload": "POST /upload-dataset",
            "generate": "POST /generate-data",
            "validate": "POST /validate-data",
            "download": "GET /download-synthetic/{dataset_id}",
            "stats": "GET /dataset-stats/{dataset_id}",
            "docs": "GET /docs",
        },
    }
