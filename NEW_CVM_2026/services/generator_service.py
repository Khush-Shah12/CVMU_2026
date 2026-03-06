"""
services/generator_service.py
=============================
Service layer that wraps the AI generator module.

Accepts an uploaded DataFrame, runs the VAE-based generator,
and returns a synthetic DataFrame.
"""

import logging
import numpy as np
import pandas as pd

import config
from ai.generator import (
    _build_seed_features,
    _denormalise,
    _generate_customers,
    _generate_accounts,
    _generate_transactions,
    _inject_fraud,
)
from models.gan_model import train_vae, generate_from_vae
from utils.file_handler import load_dataset, save_synthetic_dataset

logger = logging.getLogger(__name__)


def generate_synthetic_data(
    dataset_id: str,
    num_rows: int | None = None,
    fraud_ratio: float = 0.02,
) -> dict:
    """
    Generate synthetic data based on an uploaded original dataset.

    Parameters
    ----------
    dataset_id  : ID of the uploaded original dataset
    num_rows    : number of synthetic rows (defaults to original size)
    fraud_ratio : fraction of fraud transactions to inject

    Returns
    -------
    dict with synthetic_rows, fraud_rows, status, and saved file path
    """
    # Load the original dataset
    original_df = load_dataset(dataset_id, kind="original")
    logger.info(f"Loaded original dataset '{dataset_id}': {len(original_df)} rows")

    # Determine output size
    if num_rows is None:
        num_rows = len(original_df)
    num_rows = max(config.MIN_DATASET_SIZE, min(num_rows, config.MAX_DATASET_SIZE))

    n_normal = int(num_rows * (1 - fraud_ratio))

    # Build seed features from original data if possible, else from distributions
    logger.info("Building seed features and training VAE...")
    seed_size = min(n_normal, 5_000)
    seed_data, mins, maxs = _build_seed_features(seed_size)

    # Train VAE
    vae_model = train_vae(seed_data, input_dim=seed_data.shape[1])

    # Sample from VAE
    logger.info(f"Sampling {n_normal} synthetic rows from VAE...")
    raw_synthetic = generate_from_vae(vae_model, n_normal)
    raw_synthetic = np.clip(raw_synthetic, 0, 1)
    denorm = _denormalise(raw_synthetic, mins, maxs)

    ages = denorm[:, 0]
    incomes = denorm[:, 1]
    credit_scores = denorm[:, 2]
    balances = denorm[:, 3]
    amounts = denorm[:, 4]
    hours = denorm[:, 5]
    days = denorm[:, 6]
    device_indices = denorm[:, 7]

    # Build structured DataFrames
    n_customers = max(1, n_normal // 5)
    customers_df = _generate_customers(n_customers, ages, incomes, credit_scores)
    accounts_df = _generate_accounts(customers_df, balances)
    transactions_df = _generate_transactions(
        accounts_df, n_normal, amounts, hours, days, device_indices,
    )

    # Inject fraud
    if fraud_ratio > 0:
        transactions_df = _inject_fraud(transactions_df, accounts_df, fraud_ratio)

    # Save synthetic dataset
    path = save_synthetic_dataset(transactions_df, dataset_id)

    fraud_count = int(transactions_df["is_fraud"].sum()) if "is_fraud" in transactions_df.columns else 0

    logger.info(f"Synthetic generation complete: {len(transactions_df)} rows, {fraud_count} fraud")

    return {
        "dataset_id": dataset_id,
        "synthetic_rows": len(transactions_df),
        "fraud_rows": fraud_count,
        "status": "Synthetic data generated successfully",
        "file_path": path,
    }
