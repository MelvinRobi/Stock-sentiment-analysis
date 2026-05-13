"""Load train/test CSVs from ``archive/`` with schema checks."""

from __future__ import annotations

from typing import Optional, Tuple

import pandas as pd

from sentiment_training.config import REQUIRED_COLUMNS, TEST_CSV, TRAIN_CSV
from sentiment_training.labels import validate_binary_labels


def _assert_paths() -> None:
    if not TRAIN_CSV.is_file():
        raise FileNotFoundError(f"Missing train CSV: {TRAIN_CSV}")
    if not TEST_CSV.is_file():
        raise FileNotFoundError(f"Missing test CSV: {TEST_CSV}")


def _normalize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure sentiment is int64 and sentence is string."""
    out = df.copy()
    out["sentence"] = out["sentence"].astype("string")
    out["sentiment"] = pd.to_numeric(out["sentiment"], errors="raise").astype("int64")
    return out


def load_train(nrows: Optional[int] = None, dtype_validation: bool = True) -> pd.DataFrame:
    """
    Load ``archive/train_data.csv``.

    Parameters
    ----------
    nrows : int, optional
        Read only first N rows (recommended for development — full train is ~1.5M rows).
    dtype_validation : bool
        If True, ensure labels are only 0 and 1 (can be slow on huge frames — sample logic could be added later).
    """
    _assert_paths()
    df = pd.read_csv(TRAIN_CSV, nrows=nrows)
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"train_data.csv missing columns: {missing}")
    df = _normalize_dtypes(df)
    if dtype_validation:
        validate_binary_labels(df["sentiment"], "train.sentiment")
    return df


def load_test(dtype_validation: bool = True) -> pd.DataFrame:
    """Load full ``archive/test_data.csv`` (small file)."""
    _assert_paths()
    df = pd.read_csv(TEST_CSV)
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"test_data.csv missing columns: {missing}")
    df = _normalize_dtypes(df)
    if dtype_validation:
        validate_binary_labels(df["sentiment"], "test.sentiment")
    return df


def load_train_test(
    train_nrows: Optional[int] = None,
    validate_labels: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load both splits. Use ``train_nrows`` during iteration to avoid loading the full train set each time.
    """
    train = load_train(nrows=train_nrows, dtype_validation=validate_labels)
    test = load_test(dtype_validation=validate_labels)
    return train, test
