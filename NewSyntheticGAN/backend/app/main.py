"""
FastAPI Application Entry Point
================================
Initialises the Synthetic Financial Data Generator & Fraud Detection API.

  - Creates database tables on startup
  - Ensures upload/generated directories exist
  - Mounts all API routers
  - Configures CORS and structured logging
"""

import logging
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.database import Base, engine
from app.routers import dataset_router, model_router, synthetic_router


# ═══════════════════════════════════════════════════════════════════════════
# Logging
# ═══════════════════════════════════════════════════════════════════════════

logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Lifespan — startup / shutdown
# ═══════════════════════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Run setup logic on application startup."""
    # Create DB tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database tables ensured")

    # Create storage directories
    settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    settings.GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Upload dir : %s", settings.UPLOAD_DIR)
    logger.info("Generated dir: %s", settings.GENERATED_DIR)

    logger.info(">>  %s v%s is ready", settings.APP_TITLE, settings.APP_VERSION)
    yield
    logger.info("Shutting down ...")


# ═══════════════════════════════════════════════════════════════════════════
# FastAPI App
# ═══════════════════════════════════════════════════════════════════════════

app = FastAPI(
    title=settings.APP_TITLE,
    version=settings.APP_VERSION,
    description=(
        "AI-powered platform for generating synthetic financial transaction data "
        "and testing fraud detection models. Provides REST APIs for dataset upload, "
        "analysis, synthetic generation (SDV), model training (LR / RF / XGBoost), "
        "and dataset comparison."
    ),
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# ── CORS ────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ─────────────────────────────────────────────────────────────
app.include_router(dataset_router.router)
app.include_router(synthetic_router.router)
app.include_router(model_router.router)


# ── Health Check ────────────────────────────────────────────────────────
@app.get("/", tags=["Health"])
async def health_check():
    """Simple health-check endpoint."""
    return {
        "status": "healthy",
        "service": settings.APP_TITLE,
        "version": settings.APP_VERSION,
    }
