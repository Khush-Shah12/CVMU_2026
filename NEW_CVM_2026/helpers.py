"""
utils.py
========
Shared utility helpers used across the platform:
  - Unique ID generation
  - Realistic timestamp generation
  - Name / location sampling
  - CSV / JSON export helpers
"""

import uuid
import random
import json
import os
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd

import config


# ---------------------------------------------------------------------------
# ID generators
# ---------------------------------------------------------------------------

def generate_uuid() -> str:
    """Return a short, unique hex ID (12 chars)."""
    return uuid.uuid4().hex[:12].upper()


def generate_customer_id() -> str:
    return f"CUST-{generate_uuid()}"


def generate_account_id() -> str:
    return f"ACC-{generate_uuid()}"


def generate_transaction_id() -> str:
    return f"TXN-{generate_uuid()}"


# ---------------------------------------------------------------------------
# Name generator (simple realistic first + last combos)
# ---------------------------------------------------------------------------

FIRST_NAMES = [
    "James", "Mary", "Robert", "Patricia", "John", "Jennifer", "Michael",
    "Linda", "David", "Elizabeth", "William", "Barbara", "Richard", "Susan",
    "Joseph", "Jessica", "Thomas", "Sarah", "Christopher", "Karen",
    "Aarav", "Priya", "Hiroshi", "Yuki", "Wei", "Mei", "Aleksandr", "Olga",
    "Carlos", "Maria", "Ahmed", "Fatima", "Liam", "Emma", "Noah", "Olivia",
]

LAST_NAMES = [
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller",
    "Davis", "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez",
    "Wilson", "Anderson", "Thomas", "Taylor", "Moore", "Jackson", "Martin",
    "Patel", "Sharma", "Tanaka", "Wang", "Kim", "Nguyen", "Müller", "Ivanov",
]


def generate_name() -> str:
    """Return a random realistic full name."""
    return f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}"


# ---------------------------------------------------------------------------
# Timestamp generation
# ---------------------------------------------------------------------------

def generate_timestamps(
    n: int,
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
) -> list[datetime]:
    """
    Generate *n* random timestamps between *start* and *end*, sorted
    chronologically.  Defaults to the past 365 days.
    """
    if start is None:
        start = datetime.now() - timedelta(days=365)
    if end is None:
        end = datetime.now()
    delta = (end - start).total_seconds()
    timestamps = [start + timedelta(seconds=random.uniform(0, delta)) for _ in range(n)]
    timestamps.sort()
    return timestamps


def generate_fraud_timestamps(n: int) -> list[datetime]:
    """
    Generate timestamps concentrated in unusual hours (midnight – 5 AM).
    """
    base = datetime.now() - timedelta(days=random.randint(1, 180))
    timestamps = []
    for _ in range(n):
        hour = random.randint(*config.FRAUD_HOUR_RANGE)
        minute = random.randint(0, 59)
        second = random.randint(0, 59)
        ts = base.replace(hour=hour, minute=minute, second=second)
        timestamps.append(ts)
    timestamps.sort()
    return timestamps


# ---------------------------------------------------------------------------
# Sampling helpers
# ---------------------------------------------------------------------------

def sample_location() -> str:
    return random.choice(config.LOCATIONS)


def sample_currency() -> str:
    return random.choice(config.CURRENCIES)


def sample_device() -> str:
    return random.choice(config.DEVICE_TYPES)


def sample_account_type() -> str:
    return random.choice(config.ACCOUNT_TYPES)


def sample_transaction_type() -> str:
    return random.choice(config.TRANSACTION_TYPES)


def sample_bank_name() -> str:
    return random.choice(config.BANK_NAMES)


# ---------------------------------------------------------------------------
# Numeric distribution helpers
# ---------------------------------------------------------------------------

def clamp(value: float, lo: float, hi: float) -> float:
    """Clamp a value between lo and hi."""
    return max(lo, min(hi, value))


def sample_normal_clamped(mean: float, std: float, lo: float, hi: float) -> float:
    """Sample from a normal distribution and clamp to [lo, hi]."""
    return clamp(np.random.normal(mean, std), lo, hi)


# ---------------------------------------------------------------------------
# Export helpers
# ---------------------------------------------------------------------------

def ensure_export_dir() -> str:
    """Create the exports directory if it doesn't exist and return its path."""
    os.makedirs(config.EXPORT_DIR, exist_ok=True)
    return config.EXPORT_DIR


def export_csv(df: pd.DataFrame, filename: str) -> str:
    """Export a DataFrame to CSV in the exports directory. Returns file path."""
    path = os.path.join(ensure_export_dir(), filename)
    df.to_csv(path, index=False)
    return path


def export_json(data: list[dict] | dict, filename: str) -> str:
    """Export data (list of dicts or dict) to JSON. Returns file path."""
    path = os.path.join(ensure_export_dir(), filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)
    return path


def dataframe_to_records(df: pd.DataFrame) -> list[dict]:
    """Convert a DataFrame to a list of JSON-serialisable dicts."""
    return json.loads(df.to_json(orient="records", date_format="iso"))


def sanitise_for_json(obj):
    """
    Recursively convert numpy types to Python-native types so that
    Pydantic / json.dumps can serialise the result without errors.
    """
    if isinstance(obj, dict):
        return {k: sanitise_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitise_for_json(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj
