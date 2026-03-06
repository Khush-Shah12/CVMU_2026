"""
ai/generator.py
================
MODULE 1: Synthetic Financial Data Generator AI

Orchestrates end-to-end synthetic data generation:
  1. Build a realistic seed dataset using statistical distributions.
  2. Train a VAE on the seed data.
  3. Sample from the VAE's latent space to produce raw synthetic rows.
  4. Post-process rows into structured DataFrames:
       • customers   (customer_id, name, age, income, location, credit_score, account_type)
       • accounts    (account_id, customer_id, bank_name, account_balance, account_type)
       • transactions(transaction_id, sender_account, receiver_account, amount, currency, timestamp,
                      transaction_type, location, device_type, is_fraud)
  5. Optionally inject configurable fraud patterns.
  6. Export to CSV / JSON if requested.

All distribution parameters are pulled from config.py for easy tuning.
"""

import random
from typing import Optional

import numpy as np
import pandas as pd

import config
import helpers
from models.gan_model import train_vae, generate_from_vae


# ---------------------------------------------------------------------------
# Internal helpers for building seed data
# ---------------------------------------------------------------------------

def _build_seed_features(n: int) -> np.ndarray:
    """
    Create a (n × VAE_INPUT_DIM) normalised feature matrix that encodes:
      [age, income, credit_score, balance, amount, hour, day_of_week, device_idx]

    Values are min-max normalised to [0, 1] so the VAE can learn them.
    """
    ages = np.clip(
        np.random.normal(config.AGE_MEAN, config.AGE_STD, n),
        config.AGE_MIN, config.AGE_MAX,
    )
    incomes = np.clip(
        np.random.normal(config.INCOME_MEAN, config.INCOME_STD, n),
        config.INCOME_MIN, config.INCOME_MAX,
    )
    credit_scores = np.clip(
        np.random.normal(config.CREDIT_SCORE_MEAN, config.CREDIT_SCORE_STD, n),
        config.CREDIT_SCORE_MIN, config.CREDIT_SCORE_MAX,
    )
    balances = np.clip(incomes * np.random.uniform(0.5, 3.0, n), 1000, 1_000_000)
    amounts = np.clip(
        np.random.lognormal(
            np.log(config.TRANSACTION_AMOUNT_MEAN), config.TRANSACTION_AMOUNT_STD, n
        ),
        config.TRANSACTION_AMOUNT_MIN, config.TRANSACTION_AMOUNT_MAX,
    )
    hours = np.random.randint(0, 24, n).astype(float)
    days = np.random.randint(0, 7, n).astype(float)
    device_idx = np.random.randint(0, len(config.DEVICE_TYPES), n).astype(float)

    # Stack and min-max normalise each column to [0, 1]
    raw = np.column_stack([ages, incomes, credit_scores, balances, amounts, hours, days, device_idx])
    mins = raw.min(axis=0)
    maxs = raw.max(axis=0)
    ranges = maxs - mins
    ranges[ranges < 1e-9] = 1.0  # avoid division by zero
    normalised = (raw - mins) / ranges
    return normalised, mins, maxs


def _denormalise(data: np.ndarray, mins: np.ndarray, maxs: np.ndarray) -> np.ndarray:
    """Reverse min-max normalisation."""
    ranges = maxs - mins
    ranges[ranges < 1e-9] = 1.0
    return data * ranges + mins


# ---------------------------------------------------------------------------
# Customer profile generation
# ---------------------------------------------------------------------------

def _generate_customers(n_customers: int, ages: np.ndarray,
                        incomes: np.ndarray, credit_scores: np.ndarray) -> pd.DataFrame:
    """Create the customers DataFrame."""
    records = []
    for i in range(n_customers):
        records.append({
            "customer_id": helpers.generate_customer_id(),
            "name": helpers.generate_name(),
            "age": int(round(ages[i % len(ages)])),
            "income": round(float(incomes[i % len(incomes)]), 2),
            "location": helpers.sample_location(),
            "credit_score": int(round(credit_scores[i % len(credit_scores)])),
            "account_type": helpers.sample_account_type(),
        })
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Bank account generation
# ---------------------------------------------------------------------------

def _generate_accounts(customers: pd.DataFrame,
                       balances: np.ndarray) -> pd.DataFrame:
    """One or two accounts per customer."""
    records = []
    idx = 0
    for _, cust in customers.iterrows():
        n_accounts = random.choices([1, 2], weights=[0.6, 0.4])[0]
        for _ in range(n_accounts):
            records.append({
                "account_id": helpers.generate_account_id(),
                "customer_id": cust["customer_id"],
                "bank_name": helpers.sample_bank_name(),
                "account_balance": round(float(balances[idx % len(balances)]), 2),
                "account_type": helpers.sample_account_type(),
            })
            idx += 1
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Transaction generation
# ---------------------------------------------------------------------------

def _generate_transactions(
    accounts: pd.DataFrame,
    n_transactions: int,
    amounts: np.ndarray,
    hours: np.ndarray,
    days: np.ndarray,
    device_indices: np.ndarray,
) -> pd.DataFrame:
    """Generate normal (non-fraud) transactions."""
    account_ids = accounts["account_id"].tolist()
    timestamps = helpers.generate_timestamps(n_transactions)

    records = []
    for i in range(n_transactions):
        sender = random.choice(account_ids)
        receiver = random.choice(account_ids)
        # Ensure sender ≠ receiver
        while receiver == sender and len(account_ids) > 1:
            receiver = random.choice(account_ids)

        records.append({
            "transaction_id": helpers.generate_transaction_id(),
            "sender_account": sender,
            "receiver_account": receiver,
            "amount": round(float(amounts[i % len(amounts)]), 2),
            "currency": helpers.sample_currency(),
            "timestamp": timestamps[i].isoformat(),
            "transaction_type": helpers.sample_transaction_type(),
            "location": helpers.sample_location(),
            "device_type": config.DEVICE_TYPES[int(device_indices[i % len(device_indices)]) % len(config.DEVICE_TYPES)],
            "is_fraud": False,
        })
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Fraud injection
# ---------------------------------------------------------------------------

def _inject_fraud(transactions: pd.DataFrame, accounts: pd.DataFrame,
                  fraud_ratio: float) -> pd.DataFrame:
    """
    Generate fraudulent transactions and append them.

    Fraud types:
      1. Abnormal amount   – amount × FRAUD_AMOUNT_MULTIPLIER
      2. Unusual time      – transactions between midnight and 5 AM
      3. Repeated transfers – same sender→receiver ≥ 3 times in short window
    """
    n_fraud = max(1, int(len(transactions) * fraud_ratio))
    account_ids = accounts["account_id"].tolist()
    timestamps = helpers.generate_fraud_timestamps(n_fraud)

    records = []
    # Choose a "repeat pair" for repeated-transfer fraud
    repeat_sender = random.choice(account_ids)
    repeat_receiver = random.choice(account_ids)
    while repeat_receiver == repeat_sender and len(account_ids) > 1:
        repeat_receiver = random.choice(account_ids)

    for i in range(n_fraud):
        fraud_type = random.choice(["abnormal_amount", "unusual_time", "repeated_transfer"])

        if fraud_type == "repeated_transfer":
            sender, receiver = repeat_sender, repeat_receiver
        else:
            sender = random.choice(account_ids)
            receiver = random.choice(account_ids)
            while receiver == sender and len(account_ids) > 1:
                receiver = random.choice(account_ids)

        amount = round(
            float(np.random.lognormal(
                np.log(config.TRANSACTION_AMOUNT_MEAN),
                config.TRANSACTION_AMOUNT_STD,
            )),
            2,
        )
        if fraud_type == "abnormal_amount":
            amount = round(amount * config.FRAUD_AMOUNT_MULTIPLIER, 2)

        records.append({
            "transaction_id": helpers.generate_transaction_id(),
            "sender_account": sender,
            "receiver_account": receiver,
            "amount": amount,
            "currency": helpers.sample_currency(),
            "timestamp": timestamps[i].isoformat(),
            "transaction_type": random.choice(["transfer", "withdrawal"]),
            "location": helpers.sample_location(),
            "device_type": helpers.sample_device(),
            "is_fraud": True,
        })

    fraud_df = pd.DataFrame(records)
    combined = pd.concat([transactions, fraud_df], ignore_index=True)
    # Re-sort by timestamp to interleave fraud with normal transactions
    combined = combined.sort_values("timestamp").reset_index(drop=True)
    return combined


# ============================================================================
# PUBLIC API
# ============================================================================

def generate_synthetic_dataset(
    dataset_size: int = config.DEFAULT_DATASET_SIZE,
    fraud_ratio: float = config.DEFAULT_FRAUD_RATIO,
    export_format: Optional[str] = None,  # "csv" | "json" | None
) -> dict:
    """
    End-to-end synthetic financial data generation.

    Parameters
    ----------
    dataset_size  : total number of transactions to produce
    fraud_ratio   : fraction of transactions that are fraudulent (0.0 – 0.3)
    export_format : if set, also write files to disk

    Returns
    -------
    dict with keys:
      "customers"    : list[dict]
      "accounts"     : list[dict]
      "transactions" : list[dict]
      "summary"      : dict with counts & metadata
      "export_paths" : dict (only if export_format is set)
    """
    # Clamp inputs
    dataset_size = max(config.MIN_DATASET_SIZE, min(dataset_size, config.MAX_DATASET_SIZE))
    fraud_ratio = max(config.MIN_FRAUD_RATIO, min(fraud_ratio, config.MAX_FRAUD_RATIO))
    n_normal = int(dataset_size * (1 - fraud_ratio))

    print(f"[Generator] Generating {dataset_size} transactions (fraud ratio: {fraud_ratio:.2%}) ...")

    # ---- Step 1: Build seed data & train VAE --------------------------------
    seed_size = min(n_normal, 5_000)  # cap seed for fast training
    print(f"[Generator] Building seed data ({seed_size} rows) ...")
    seed_data, mins, maxs = _build_seed_features(seed_size)

    print("[Generator] Training VAE ...")
    vae_model = train_vae(seed_data, input_dim=seed_data.shape[1])

    # ---- Step 2: Sample from VAE --------------------------------------------
    print(f"[Generator] Sampling {n_normal} synthetic rows from VAE ...")
    raw_synthetic = generate_from_vae(vae_model, n_normal)
    # Clip to [0, 1] (VAE sigmoid should do this but be safe)
    raw_synthetic = np.clip(raw_synthetic, 0, 1)
    # Denormalise
    denorm = _denormalise(raw_synthetic, mins, maxs)

    ages = denorm[:, 0]
    incomes = denorm[:, 1]
    credit_scores = denorm[:, 2]
    balances = denorm[:, 3]
    amounts = denorm[:, 4]
    hours = denorm[:, 5]
    days = denorm[:, 6]
    device_indices = denorm[:, 7]

    # ---- Step 3: Build structured DataFrames ---------------------------------
    n_customers = max(1, n_normal // 5)  # ~5 txns per customer
    print(f"[Generator] Creating {n_customers} customer profiles ...")
    customers_df = _generate_customers(n_customers, ages, incomes, credit_scores)

    print("[Generator] Creating bank accounts ...")
    accounts_df = _generate_accounts(customers_df, balances)

    print(f"[Generator] Creating {n_normal} normal transactions ...")
    transactions_df = _generate_transactions(
        accounts_df, n_normal, amounts, hours, days, device_indices,
    )

    # ---- Step 4: Inject fraud ------------------------------------------------
    if fraud_ratio > 0:
        n_fraud = dataset_size - n_normal
        print(f"[Generator] Injecting {n_fraud} fraud transactions ...")
        transactions_df = _inject_fraud(transactions_df, accounts_df, fraud_ratio)

    # ---- Step 5: Export (optional) -------------------------------------------
    export_paths = {}
    if export_format == "csv":
        export_paths["customers"] = helpers.export_csv(customers_df, "customers.csv")
        export_paths["accounts"] = helpers.export_csv(accounts_df, "accounts.csv")
        export_paths["transactions"] = helpers.export_csv(transactions_df, "transactions.csv")
    elif export_format == "json":
        export_paths["customers"] = helpers.export_json(helpers.dataframe_to_records(customers_df), "customers.json")
        export_paths["accounts"] = helpers.export_json(helpers.dataframe_to_records(accounts_df), "accounts.json")
        export_paths["transactions"] = helpers.export_json(helpers.dataframe_to_records(transactions_df), "transactions.json")

    print("[Generator] [DONE] Generation complete.")

    summary = {
        "total_customers": len(customers_df),
        "total_accounts": len(accounts_df),
        "total_transactions": len(transactions_df),
        "fraud_transactions": int(transactions_df["is_fraud"].sum()),
        "fraud_ratio_actual": round(float(transactions_df["is_fraud"].mean()), 4),
        "dataset_size_requested": dataset_size,
    }

    result = {
        "customers": helpers.dataframe_to_records(customers_df),
        "accounts": helpers.dataframe_to_records(accounts_df),
        "transactions": helpers.dataframe_to_records(transactions_df),
        "summary": summary,
    }
    if export_paths:
        result["export_paths"] = export_paths

    return result
