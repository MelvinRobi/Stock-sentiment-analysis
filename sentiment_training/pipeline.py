"""
Glue preprocessing steps for model-ready DataFrames.

Typical flow::

    from sentiment_training.io import load_train_test
    from sentiment_training.pipeline import prepare_tokenized_splits

    train, test = load_train_test(train_nrows=50_000)
    train_tok, test_tok = prepare_tokenized_splits(train, test)
"""

from __future__ import annotations

from typing import Tuple

import pandas as pd

from sentiment_training.preprocess import apply_clean_tokenize


def prepare_tokenized_splits(
    train: pd.DataFrame,
    test: pd.DataFrame,
    text_col: str = "sentence",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Return copies with ``text_clean`` and ``tokens`` columns added.

    Keeps original ``sentence`` and ``sentiment`` untouched.
    """
    train_out = apply_clean_tokenize(train, text_col=text_col)
    test_out = apply_clean_tokenize(test, text_col=text_col)
    return train_out, test_out
