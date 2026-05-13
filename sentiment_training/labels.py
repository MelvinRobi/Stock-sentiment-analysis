"""
Label schema for `archive/train_data.csv` and `archive/test_data.csv`.

Verified encoding: integer **binary** only — not three-way positive/neutral/negative.
"""

from __future__ import annotations

import pandas as pd

# Numeric codes in the CSV files
NEGATIVE = 0
POSITIVE = 1

ALLOWED_BINARY = frozenset({NEGATIVE, POSITIVE})

# Human-readable names for modeling / reporting
NAME_BY_ID: dict[int, str] = {
    NEGATIVE: "negative",
    POSITIVE: "positive",
}


def validate_binary_labels(series: pd.Series, name: str = "sentiment") -> None:
    """Raise ValueError if any label is outside {0, 1}."""
    bad = ~series.isin(list(ALLOWED_BINARY))
    if bad.any():
        extras = sorted(series.loc[bad].unique().tolist())
        raise ValueError(f"{name}: unexpected labels {extras}; expected only {sorted(ALLOWED_BINARY)}")


def add_label_name(df: pd.DataFrame, col: str = "sentiment", out: str = "label_name") -> pd.DataFrame:
    """Append string column `negative` / `positive` for convenience."""
    out_df = df.copy()
    out_df[out] = out_df[col].map(NAME_BY_ID)
    return out_df
