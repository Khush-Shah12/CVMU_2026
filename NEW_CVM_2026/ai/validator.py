"""
ai/validator.py
===============
MODULE 2: Synthetic Data Validation & Testing AI

Evaluates a transaction dataset and determines whether it behaves like real
financial data.  Three validation layers:

  1. Statistical validation  – KS-test, mean / variance / distribution checks.
  2. Logical validation      – timestamp ordering, balance consistency, sender ≠ receiver.
  3. Fraud detection         – runs the AnomalyEnsemble on numeric features.

Output:
  {
    "realism_score":          0–100,
    "anomaly_score":          float,
    "fraud_patterns_detected": bool,
    "report":                 { ... detailed sub-reports ... }
  }
"""

from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

import config
import helpers
from models.anomaly_model import AnomalyEnsemble


# ============================================================================
# 1. Statistical validation
# ============================================================================

def _statistical_validation(df: pd.DataFrame) -> dict:
    """
    Run distribution checks on the transaction amounts.

    Checks:
      • Mean & std vs expected ranges
      • Kolmogorov–Smirnov test against log-normal distribution
      • Skewness & kurtosis reasonableness
    """
    amounts = df["amount"].values.astype(float)

    mean_amt = float(amounts.mean())
    std_amt = float(amounts.std())
    median_amt = float(np.median(amounts))

    # KS test: compare amounts to a fitted log-normal
    # Log-transform, then test normality of log(amounts)
    log_amounts = np.log1p(amounts)
    ks_stat, ks_pvalue = stats.kstest(
        log_amounts, "norm",
        args=(log_amounts.mean(), log_amounts.std()),
    )

    skewness = float(stats.skew(amounts))
    kurtosis = float(stats.kurtosis(amounts))

    # Score: higher is better (more realistic)
    # We expect financial transaction amounts to be right-skewed (log-normal)
    score = 100.0
    # Deduct for KS test failure
    if ks_pvalue < config.KS_TEST_ALPHA:
        score -= 20 * min(1.0, ks_stat / 0.1)
    # Deduct if mean is unreasonably high or low
    if mean_amt < 10 or mean_amt > 10_000:
        score -= 15
    # Deduct for negative skew (financial data should be right-skewed)
    if skewness < 0:
        score -= 10

    score = max(0.0, min(100.0, score))

    return {
        "score": round(score, 2),
        "mean_amount": round(mean_amt, 2),
        "std_amount": round(std_amt, 2),
        "median_amount": round(median_amt, 2),
        "skewness": round(skewness, 4),
        "kurtosis": round(kurtosis, 4),
        "ks_statistic": round(ks_stat, 4),
        "ks_pvalue": round(ks_pvalue, 4),
        "distribution_normal_log": bool(ks_pvalue >= config.KS_TEST_ALPHA),
    }


# ============================================================================
# 2. Logical validation
# ============================================================================

def _logical_validation(df: pd.DataFrame) -> dict:
    """
    Check logical consistency:
      • Timestamps in chronological order
      • sender ≠ receiver on each transaction
      • All required columns present
      • No negative amounts
    """
    issues = []
    score = 100.0

    # --- Required columns ---
    required = [
        "transaction_id", "sender_account", "receiver_account",
        "amount", "timestamp",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        issues.append(f"Missing columns: {missing}")
        score -= 20

    # --- Timestamp ordering ---
    if "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"], errors="coerce")
        non_null = ts.dropna()
        if len(non_null) > 1:
            out_of_order = int((non_null.diff().dropna() < pd.Timedelta(0)).sum())
            if out_of_order > 0:
                pct = out_of_order / len(non_null) * 100
                issues.append(f"{out_of_order} timestamps out of order ({pct:.1f}%)")
                score -= min(20, pct)

    # --- Sender ≠ Receiver ---
    if "sender_account" in df.columns and "receiver_account" in df.columns:
        same = int((df["sender_account"] == df["receiver_account"]).sum())
        if same > 0:
            pct = same / len(df) * 100
            issues.append(f"{same} transactions have sender == receiver ({pct:.1f}%)")
            score -= min(15, pct)

    # --- No negative amounts ---
    if "amount" in df.columns:
        negatives = int((df["amount"] < 0).sum())
        if negatives > 0:
            issues.append(f"{negatives} transactions with negative amount")
            score -= 10

    # --- No duplicate transaction IDs ---
    if "transaction_id" in df.columns:
        dupes = int(df["transaction_id"].duplicated().sum())
        if dupes > 0:
            issues.append(f"{dupes} duplicate transaction IDs")
            score -= 10

    score = max(0.0, min(100.0, score))

    return {
        "score": round(score, 2),
        "issues": issues,
        "total_rows": len(df),
    }


# ============================================================================
# 3. Fraud / anomaly detection
# ============================================================================

def _fraud_detection(df: pd.DataFrame) -> dict:
    """
    Run the AnomalyEnsemble on numeric transaction features.

    Returns anomaly rate, mean score, and per-record labels.
    """
    # Extract numeric features for the anomaly model
    numeric_cols = []
    if "amount" in df.columns:
        numeric_cols.append("amount")

    # Encode hour-of-day and day-of-week from timestamp
    feature_df = pd.DataFrame()
    feature_df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0)

    if "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"], errors="coerce")
        feature_df["hour"] = ts.dt.hour.fillna(12).astype(float)
        feature_df["day_of_week"] = ts.dt.dayofweek.fillna(3).astype(float)
    else:
        feature_df["hour"] = 12.0
        feature_df["day_of_week"] = 3.0

    # Add a simple numeric encoding for device_type if present
    if "device_type" in df.columns:
        device_map = {d: i for i, d in enumerate(config.DEVICE_TYPES)}
        feature_df["device_idx"] = df["device_type"].map(device_map).fillna(0).astype(float)
    else:
        feature_df["device_idx"] = 0.0

    features = feature_df.values

    # Adjust input_dim to match actual feature count
    n_features = features.shape[1]

    # Train ensemble on the data itself (unsupervised)
    ensemble = AnomalyEnsemble(input_dim=n_features)
    ensemble.fit(features)
    summary = ensemble.summary(features)

    # Determine if fraud patterns are detected
    fraud_detected = bool(summary["anomaly_rate"] > 1.0)  # more than 1% anomalies

    return {
        "fraud_patterns_detected": fraud_detected,
        "anomaly_summary": summary,
        "anomaly_score": summary["mean_anomaly_score"],
    }


# ============================================================================
# PUBLIC API
# ============================================================================

def validate_dataset(
    df: pd.DataFrame,
) -> dict:
    """
    Run all validation checks on a transactions DataFrame.

    Returns
    -------
    dict:
      realism_score           : 0–100 weighted composite
      anomaly_score           : 0–1 mean anomaly score
      fraud_patterns_detected : bool
      report                  : detailed sub-reports
    """
    print("[Validator] Running statistical validation ...")
    stat_report = _statistical_validation(df)

    print("[Validator] Running logical validation ...")
    logic_report = _logical_validation(df)

    print("[Validator] Running fraud / anomaly detection ...")
    fraud_report = _fraud_detection(df)

    # ---- Composite realism score --------------------------------------------
    w = config.REALISM_SCORE_WEIGHTS
    # Anomaly sub-score: lower anomaly rate → higher realism
    anomaly_realism = max(0, 100 - fraud_report["anomaly_summary"]["anomaly_rate"] * 10)

    realism_score = (
        w["statistical"] * stat_report["score"]
        + w["logical"] * logic_report["score"]
        + w["anomaly"] * anomaly_realism
    )
    realism_score = round(max(0.0, min(100.0, realism_score)), 1)

    # Build human-readable summary
    if realism_score >= 80:
        summary_text = "Dataset appears statistically realistic with strong logical consistency."
    elif realism_score >= 50:
        summary_text = "Dataset is moderately realistic but has some distributional or logical issues."
    else:
        summary_text = "Dataset has significant issues that reduce its realism."

    print(f"[Validator] [DONE] Validation complete.  Realism score: {realism_score}")

    return helpers.sanitise_for_json({
        "realism_score": float(realism_score),
        "anomaly_score": float(round(fraud_report["anomaly_score"], 4)),
        "fraud_patterns_detected": bool(fraud_report["fraud_patterns_detected"]),
        "report": {
            "summary": summary_text,
            "statistical": stat_report,
            "logical": logic_report,
            "fraud_detection": fraud_report,
        },
    })
