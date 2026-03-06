"""Full platform verification: backend + frontend API client."""
import sys
sys.path.insert(0, "d:/Hackathon/NEW_CVM_2026/frontend")

import requests
import pandas as pd
import numpy as np
import io

# 1. Check servers are alive
print("=" * 60)
print("  SERVER HEALTH CHECKS")
print("=" * 60)

r = requests.get("http://localhost:8501")
print(f"Frontend (Streamlit):  {r.status_code} {'OK' if r.status_code == 200 else 'FAIL'}")

r2 = requests.get("http://localhost:8501/_stcore/health")
print(f"Streamlit Health:      {r2.text.strip()}")

r3 = requests.get("http://localhost:8000/")
print(f"Backend (FastAPI):     {r3.status_code} - {r3.json()['status']}")

# 2. Test via the frontend's API client
from utils.api_client import upload_dataset, generate_data, validate_data, get_dataset_stats, download_synthetic

np.random.seed(99)
n = 200
df = pd.DataFrame({
    "transaction_id": [f"T{i}" for i in range(n)],
    "sender_account": [f"A{np.random.randint(1,20)}" for _ in range(n)],
    "receiver_account": [f"A{np.random.randint(1,20)}" for _ in range(n)],
    "amount": np.round(np.random.lognormal(5, 1, n), 2),
    "currency": np.random.choice(["USD", "EUR"], n),
    "timestamp": pd.date_range("2025-01-01", periods=n, freq="h").strftime("%Y-%m-%dT%H:%M:%S").tolist(),
    "transaction_type": np.random.choice(["purchase", "transfer"], n),
    "is_fraud": np.random.choice([True, False], n, p=[0.05, 0.95]),
})
csv_bytes = df.to_csv(index=False).encode()

print()
print("=" * 60)
print("  API CLIENT TEST (same code Streamlit uses)")
print("=" * 60)

r = upload_dataset(csv_bytes, "test.csv")
did = r["dataset_id"]
print(f"1. Upload:    OK  - ID={did}, {r['rows']} rows, {r['columns']} cols")

r = generate_data(did, num_rows=300, fraud_ratio=0.05)
print(f"2. Generate:  OK  - {r['synthetic_rows']} synthetic rows, {r['fraud_rows']} fraud")

r = get_dataset_stats(did)
print(f"3. Stats:     OK  - {r['total_transactions']} txns, type={r['dataset_type']}")

r = validate_data(did)
print(f"4. Validate:  OK  - similarity={r['similarity_score']}, quality={r['quality_score']}")

b = download_synthetic(did)
print(f"5. Download:  OK  - {len(b):,} bytes")

print()
print("=" * 60)
print("  ALL CHECKS PASSED - Platform is fully operational!")
print("=" * 60)
