"""
config.py
=========
Central configuration constants for the Finance Synthetic Data Generator & Validator Platform.

All tuneable hyper-parameters, default values, and shared constants live here so
they can be adjusted from a single location without touching business logic.
"""

# ---------------------------------------------------------------------------
# Dataset generation defaults
# ---------------------------------------------------------------------------
DEFAULT_DATASET_SIZE: int = 10_000          # rows of transactions to generate
MIN_DATASET_SIZE: int = 100
MAX_DATASET_SIZE: int = 5_000_000           # upper cap for safety
DEFAULT_FRAUD_RATIO: float = 0.02           # 2 % fraud by default
MIN_FRAUD_RATIO: float = 0.0
MAX_FRAUD_RATIO: float = 0.30              # cap at 30 %

# ---------------------------------------------------------------------------
# Customer profile distributions
# ---------------------------------------------------------------------------
AGE_MEAN: float = 42.0
AGE_STD: float = 14.0
AGE_MIN: int = 18
AGE_MAX: int = 85

INCOME_MEAN: float = 55_000.0
INCOME_STD: float = 30_000.0
INCOME_MIN: float = 12_000.0
INCOME_MAX: float = 500_000.0

CREDIT_SCORE_MEAN: float = 680.0
CREDIT_SCORE_STD: float = 80.0
CREDIT_SCORE_MIN: int = 300
CREDIT_SCORE_MAX: int = 850

# ---------------------------------------------------------------------------
# Transaction distributions
# ---------------------------------------------------------------------------
TRANSACTION_AMOUNT_MEAN: float = 250.0      # log-normal location param
TRANSACTION_AMOUNT_STD: float = 1.2         # log-normal scale param
TRANSACTION_AMOUNT_MIN: float = 0.50
TRANSACTION_AMOUNT_MAX: float = 50_000.0

# Fraud-specific overrides
FRAUD_AMOUNT_MULTIPLIER: float = 8.0        # normal amount × this
FRAUD_HOUR_RANGE: tuple = (0, 5)            # unusual hours (midnight–5 AM)
FRAUD_REPEAT_MIN: int = 3                   # min repeated transfers to flag

# ---------------------------------------------------------------------------
# Reference pools (used by utils / generator)
# ---------------------------------------------------------------------------
CURRENCIES: list[str] = ["USD", "EUR", "GBP", "INR", "JPY", "CAD", "AUD"]
LOCATIONS: list[str] = [
    "New York", "London", "Mumbai", "Tokyo", "Toronto",
    "Sydney", "Berlin", "Singapore", "Dubai", "San Francisco",
    "Chicago", "Paris", "Hong Kong", "São Paulo", "Seoul",
]
DEVICE_TYPES: list[str] = ["mobile", "desktop", "tablet", "ATM", "POS"]
ACCOUNT_TYPES: list[str] = ["savings", "checking", "business", "credit"]
TRANSACTION_TYPES: list[str] = [
    "purchase", "transfer", "withdrawal", "deposit", "payment", "refund",
]
BANK_NAMES: list[str] = [
    "National Bank", "Global Trust", "City Finance", "Metro Credit Union",
    "Alliance Savings", "Pacific Bank", "Heritage Financial", "United Capital",
]

# ---------------------------------------------------------------------------
# VAE hyper-parameters
# ---------------------------------------------------------------------------
VAE_INPUT_DIM: int = 8                      # number of numeric features fed to VAE
VAE_HIDDEN_DIM: int = 64
VAE_LATENT_DIM: int = 16
VAE_LEARNING_RATE: float = 1e-3
VAE_EPOCHS: int = 50
VAE_BATCH_SIZE: int = 256

# ---------------------------------------------------------------------------
# Anomaly detection hyper-parameters
# ---------------------------------------------------------------------------
ISOLATION_FOREST_CONTAMINATION: float = 0.05
ISOLATION_FOREST_N_ESTIMATORS: int = 200
AUTOENCODER_HIDDEN_DIM: int = 32
AUTOENCODER_LATENT_DIM: int = 8
AUTOENCODER_EPOCHS: int = 30
AUTOENCODER_LR: float = 1e-3
AUTOENCODER_BATCH_SIZE: int = 256
ANOMALY_ENSEMBLE_IF_WEIGHT: float = 0.5     # weight for Isolation Forest score
ANOMALY_ENSEMBLE_AE_WEIGHT: float = 0.5     # weight for Autoencoder score

# ---------------------------------------------------------------------------
# Validation thresholds
# ---------------------------------------------------------------------------
KS_TEST_ALPHA: float = 0.05                 # significance level for KS test
REALISM_SCORE_WEIGHTS: dict = {
    "statistical": 0.40,
    "logical": 0.35,
    "anomaly": 0.25,
}

# ---------------------------------------------------------------------------
# File / export settings
# ---------------------------------------------------------------------------
EXPORT_DIR: str = "exports"

# ---------------------------------------------------------------------------
# File upload & dataset storage settings
# ---------------------------------------------------------------------------
MAX_FILE_SIZE_MB: float = 100.0              # max upload size in MB
ORIGINAL_DATASET_DIR: str = "datasets/original"
SYNTHETIC_DATASET_DIR: str = "datasets/synthetic"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_LEVEL: str = "INFO"
LOG_FORMAT: str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"

