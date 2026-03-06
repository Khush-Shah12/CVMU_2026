"""
Application configuration using pydantic-settings.
Reads from environment variables and .env file.
"""

import os
from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Global application settings."""

    # ── Project Paths ───────────────────────────────────────────────────
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    UPLOAD_DIR: Path = Path(__file__).resolve().parent.parent / "uploads"
    GENERATED_DIR: Path = Path(__file__).resolve().parent.parent / "generated"

    # ── Database ────────────────────────────────────────────────────────
    DATABASE_URL: str = "postgresql+asyncpg://postgres:Khush%402412@localhost:5432/syntheticgan"

    # ── Logging ─────────────────────────────────────────────────────────
    LOG_LEVEL: str = "INFO"

    # ── App Meta ────────────────────────────────────────────────────────
    APP_TITLE: str = "Synthetic Financial Data Generator & Fraud Detection API"
    APP_VERSION: str = "1.0.0"

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


# Singleton instance
settings = Settings()
