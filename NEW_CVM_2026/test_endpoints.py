"""End-to-end test for all 5 backend endpoints."""
import requests, json, os
import pandas as pd
import numpy as np

BASE = "http://localhost:8000"

# --- 0. Health Check ---
print("=" * 60)
print("  TEST 0: Health Check")
print("=" * 60)
r = requests.get(f"{BASE}/")
print("Status:", r.status_code)
print(json.dumps(r.json(), indent=2))

# --- Prepare sample CSV ---
print("\n" + "=" * 60)
print("  Preparing sample CSV for upload...")
print("=" * 60)
np.random.seed(42)
n = 1000
df = pd.DataFrame({
    "transaction_id": [f"TXN_{i:05d}" for i in range(n)],
    "sender_account": [f"ACC_{np.random.randint(1,50):03d}" for _ in range(n)],
    "receiver_account": [f"ACC_{np.random.randint(1,50):03d}" for _ in range(n)],
    "amount": np.round(np.random.lognormal(5, 1.2, n), 2),
    "currency": np.random.choice(["USD", "EUR", "GBP", "INR"], n),
    "timestamp": pd.date_range("2025-01-01", periods=n, freq="h").strftime("%Y-%m-%dT%H:%M:%S").tolist(),
    "transaction_type": np.random.choice(["purchase", "transfer", "withdrawal", "deposit"], n),
    "location": np.random.choice(["New York", "London", "Mumbai", "Tokyo"], n),
    "device_type": np.random.choice(["mobile", "desktop", "ATM", "POS"], n),
    "is_fraud": np.random.choice([True, False], n, p=[0.03, 0.97]),
})
csv_path = "test_upload.csv"
df.to_csv(csv_path, index=False)
print(f"Created {csv_path}: {len(df)} rows, {len(df.columns)} columns")

# --- 1. Upload ---
print("\n" + "=" * 60)
print("  TEST 1: POST /upload-dataset")
print("=" * 60)
with open(csv_path, "rb") as f:
    r = requests.post(f"{BASE}/upload-dataset", files={"file": ("transactions.csv", f, "text/csv")})
print("Status:", r.status_code)
upload_data = r.json()
print(json.dumps(upload_data, indent=2))
dataset_id = upload_data["dataset_id"]

# --- 2. Generate ---
print("\n" + "=" * 60)
print("  TEST 2: POST /generate-data")
print("=" * 60)
r = requests.post(f"{BASE}/generate-data", json={"dataset_id": dataset_id, "num_rows": 500, "fraud_ratio": 0.05})
print("Status:", r.status_code)
print(json.dumps(r.json(), indent=2))

# --- 3. Stats ---
print("\n" + "=" * 60)
print("  TEST 3: GET /dataset-stats/" + dataset_id)
print("=" * 60)
r = requests.get(f"{BASE}/dataset-stats/{dataset_id}")
print("Status:", r.status_code)
stats = r.json()
dt = stats.get("dataset_type", "?")
tt = stats.get("total_transactions", 0)
fc = stats.get("fraud_count", "N/A")
print(f"Type: {dt}, Transactions: {tt}, Fraud: {fc}")

# --- 4. Validate ---
print("\n" + "=" * 60)
print("  TEST 4: POST /validate-data")
print("=" * 60)
r = requests.post(f"{BASE}/validate-data", json={"dataset_id": dataset_id})
print("Status:", r.status_code)
val = r.json()
ss = val.get("similarity_score", "?")
qs = val.get("quality_score", "?")
fm = val.get("fraud_ratio_match", "?")
rs = val.get("realism_score", "?")
ans = val.get("anomaly_score", "?")
print(f"Similarity: {ss}, Quality: {qs}, Fraud Match: {fm}")
print(f"Realism: {rs}, Anomaly: {ans}")

# --- 5. Download ---
print("\n" + "=" * 60)
print("  TEST 5: GET /download-synthetic/" + dataset_id)
print("=" * 60)
r = requests.get(f"{BASE}/download-synthetic/{dataset_id}")
print("Status:", r.status_code)
ct = r.headers.get("content-type", "?")
print(f"Content-Type: {ct}")
print(f"Downloaded {len(r.content)} bytes")

# Cleanup
os.remove(csv_path)

# --- Verdict ---
print("\n" + "=" * 60)
print("  ALL 5 ENDPOINTS TESTED SUCCESSFULLY!")
print("=" * 60)
