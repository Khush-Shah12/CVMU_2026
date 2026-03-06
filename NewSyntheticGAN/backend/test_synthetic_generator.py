import os
from pathlib import Path

import pandas as pd
import pytest

from app.services.synthetic_generator import (
    balance_classes,
    generate_synthetic_data,
    train_synthesizer,
)


def _make_imbalanced_dataframe() -> pd.DataFrame:
    """Helper to build a tiny imbalanced binary dataset."""
    data = {
        "amount": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        "fraud": [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    }
    return pd.DataFrame(data)


def test_balance_classes_balances_majority_and_minority():
    df = _make_imbalanced_dataframe()
    balanced = balance_classes(df, "fraud")

    counts = balanced["fraud"].value_counts()
    assert set(counts.index.tolist()) == {0, 1}
    # After balancing, the two classes should have equal counts
    assert counts[0] == counts[1]


def test_balance_classes_single_class_returns_unchanged():
    df = pd.DataFrame({"fraud": [0, 0, 0, 0]})
    balanced = balance_classes(df, "fraud")
    pd.testing.assert_frame_equal(df, balanced)


def test_balance_classes_missing_column_raises():
    df = pd.DataFrame({"x": [1, 2, 3]})
    with pytest.raises(ValueError):
        balance_classes(df, "fraud")


def test_generate_raises_if_not_trained():
    # The global synthesizer should not be initialised yet in a fresh test run.
    with pytest.raises(RuntimeError):
        generate_synthetic_data(num_rows=10)


def test_train_and_generate_roundtrip(tmp_path: Path):
    df = _make_imbalanced_dataframe()

    # Train without balancing
    meta = train_synthesizer(df, target_column="fraud", balance=False)
    assert meta["rows_trained"] == len(df)
    assert meta["balanced"] is False
    assert meta["target_column"] == "fraud"
    assert isinstance(meta["run_id"], str)

    # Generate a small synthetic sample, saving to disk
    out_path = tmp_path / "synthetic.csv"
    synthetic_df, gen_meta = generate_synthetic_data(
        num_rows=25,
        output_path=out_path,
        random_state=123,
    )

    assert len(synthetic_df) == 25
    assert gen_meta["generated_rows"] == 25
    assert gen_meta["saved_to"] is not None
    assert Path(gen_meta["saved_to"]).exists()
    assert set(gen_meta["columns"]) == set(synthetic_df.columns)
    assert gen_meta["training_meta"]["run_id"] == meta["run_id"]


def test_train_with_balancing_upsamples_minority():
    df = _make_imbalanced_dataframe()
    original_count = len(df)

    meta = train_synthesizer(df, target_column="fraud", balance=True)

    # When balancing is enabled, the number of rows used for training
    # should be greater than or equal to the original.
    assert meta["balanced"] is True
    assert meta["rows_trained"] >= original_count

