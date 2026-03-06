import time
from pathlib import Path
from threading import Lock
from typing import Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd
from faker import Faker
from sdv.metadata import SingleTableMetadata
from sdv.single_table import GaussianCopulaSynthesizer
from sklearn.utils import resample

faker = Faker()
_lock = Lock()
_synthesizer: Optional[GaussianCopulaSynthesizer] = None
_table_metadata: Optional[SingleTableMetadata] = None
_last_training_meta: Dict[str, Any] = {}


def balance_classes(df: pd.DataFrame, target_column: str) -> pd.DataFrame:
    """Return an upsampled, class-balanced copy of df for the given binary target."""
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found.")
    class_counts = df[target_column].value_counts()
    if class_counts.nunique() == 1:
        return df.copy()

    majority = class_counts.idxmax()
    minority = class_counts.idxmin()
    df_major = df[df[target_column] == majority]
    df_minor = df[df[target_column] == minority]

    df_minor_up = resample(
        df_minor,
        replace=True,
        n_samples=len(df_major),
        random_state=42,
    )
    balanced = pd.concat([df_major, df_minor_up], axis=0).sample(frac=1, random_state=42)
    return balanced.reset_index(drop=True)


def train_synthesizer(
    dataframe: pd.DataFrame,
    *,
    target_column: Optional[str] = None,
    balance: bool = False,
) -> Dict[str, Any]:
    """Fit a GaussianCopulaSynthesizer on the provided dataframe."""
    global _synthesizer, _table_metadata, _last_training_meta

    if dataframe.empty:
        raise ValueError("Input dataframe is empty.")

    df_train = dataframe.copy()
    if balance and target_column:
        df_train = balance_classes(df_train, target_column)

    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(data=df_train)

    start = time.time()
    synth = GaussianCopulaSynthesizer(metadata)
    synth.fit(df_train)
    elapsed = time.time() - start

    with _lock:
        _synthesizer = synth
        _table_metadata = metadata
        _last_training_meta = {
            "run_id": faker.uuid4(),
            "rows_trained": int(len(df_train)),
            "columns": list(df_train.columns),
            "balanced": bool(balance and target_column),
            "target_column": target_column,
            "elapsed_seconds": round(elapsed, 4),
            "field_types": metadata.to_dict().get("fields", {}),
        }
    return _last_training_meta.copy()


def generate_synthetic_data(
    num_rows: int,
    *,
    output_path: Optional[str | Path] = None,
    random_state: Optional[int] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Generate synthetic rows; optionally persist to CSV; return data + metadata."""
    if _synthesizer is None:
        raise RuntimeError("Synthesizer not trained. Call train_synthesizer first.")
    if num_rows <= 0:
        raise ValueError("num_rows must be positive.")

    start = time.time()
    synthetic_df = _synthesizer.sample(num_rows=num_rows, random_state=random_state)
    elapsed = time.time() - start

    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        synthetic_df.to_csv(path, index=False)

    meta = {
        "run_id": faker.uuid4(),
        "generated_rows": int(num_rows),
        "saved_to": str(output_path) if output_path else None,
        "elapsed_seconds": round(elapsed, 4),
        "columns": list(synthetic_df.columns),
        "training_meta": _last_training_meta.copy(),
    }
    return synthetic_df, meta
