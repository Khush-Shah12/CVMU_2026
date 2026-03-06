"""
Utility helpers for data cleaning and preprocessing.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def clean_dataframe(
    df: pd.DataFrame,
    drop_duplicates: bool = True,
    fill_strategy: str = "median",
) -> pd.DataFrame:
    """
    Clean a raw DataFrame:
      1. Strip whitespace from string columns
      2. Drop full-duplicate rows (optional)
      3. Coerce numeric-looking columns to numeric types
      4. Fill missing values using the chosen strategy
    """
    df = df.copy()

    # ── Strip whitespace ────────────────────────────────────────────────
    str_cols = df.select_dtypes(include=["object"]).columns
    for col in str_cols:
        df[col] = df[col].astype(str).str.strip()
        # Replace 'nan' strings that arose from casting
        df[col] = df[col].replace("nan", np.nan)

    # ── Drop duplicates ─────────────────────────────────────────────────
    if drop_duplicates:
        before = len(df)
        df = df.drop_duplicates()
        dropped = before - len(df)
        if dropped:
            logger.info("Dropped %d duplicate rows", dropped)

    # ── Coerce numeric columns ──────────────────────────────────────────
    for col in df.columns:
        if df[col].dtype == object:
            try:
                converted = pd.to_numeric(df[col], errors="coerce")
                # Only convert if >50 % of values parsed successfully
                if converted.notna().mean() > 0.5:
                    df[col] = converted
            except Exception:
                pass

    # ── Fill missing values ─────────────────────────────────────────────
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(include=["object"]).columns

    if fill_strategy == "median":
        for col in numeric_cols:
            if df[col].isna().any():
                df[col] = df[col].fillna(df[col].median())
    elif fill_strategy == "mean":
        for col in numeric_cols:
            if df[col].isna().any():
                df[col] = df[col].fillna(df[col].mean())
    elif fill_strategy == "zero":
        df[numeric_cols] = df[numeric_cols].fillna(0)

    for col in categorical_cols:
        if df[col].isna().any():
            df[col] = df[col].fillna("Unknown")

    logger.info("Cleaning complete - %d rows x %d cols", *df.shape)
    return df


def detect_outliers_iqr(
    series: pd.Series, factor: float = 1.5
) -> pd.Series:
    """Return boolean mask of outliers using the IQR method."""
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - factor * iqr
    upper = q3 + factor * iqr
    return (series < lower) | (series > upper)
