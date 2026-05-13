"""
Normalize arbitrary CSV exports into a single schema for sentiment analysis.

Expected output columns (internal standard):
  - date: optional datetime (NaT if missing)
  - tweet: primary text to score (from a text column or combined string fields)
  - ticker, company: optional grouping metadata

Adding a Kaggle dataset (fresh workflow):
    1. Download your CSV from Kaggle.
    2. Save it under ``data/kaggle/your_file.csv`` (folder is created for you).
    3. Restart the app, pick the file in the Dataset dropdown, click Run analysis.

Root-level ``*.csv`` files are still discovered. Use ``data/kaggle/`` to keep imports organized.
Training splits under ``archive/*.csv`` (e.g. ``train_data.csv``) are listed in the same dropdown.

To tune column detection for an unusual export, edit ``TEXT_COLUMN_CANDIDATES`` or
``COMBINE_COLUMN_BLOCK`` in this module.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

from sentiment_dashboard.config import ARCHIVE_DIR, DATA_KAGGLE_DIR, PRIORITY_DEFAULT_CSV, PROJECT_ROOT

# Ordered: first exact/substring match wins for primary text
TEXT_COLUMN_CANDIDATES = [
    "tweet",
    "sentence",
    "text",
    "content",
    "message",
    "body",
    "review",
    "comment",
    "post",
    "title",
    "description",
    "feedback",
    "response",
    "answer",
    "transcript",
    "note",
    "summary",
]

ID_LIKE_COLUMN = re.compile(r"^(?:.*_)?id$|^index$|^idx$|^row$", re.I)
COMBINE_COLUMN_BLOCK = re.compile(
    r"gender|popularity|student_?id|_?age$|^age$|gpa|frequency|hours|weekly|baseline|post_ai|"
    r"confidence|time_saved|ethics_?concern|task_?frequency|department_?id|user_?id|lat|lon|zip",
    re.I,
)
NUMERIC_STRING = re.compile(r"^-?\d+(\.\d+)?$")


def discover_csv_filenames() -> list[str]:
    """
    Discover CSV datasets as paths relative to ``PROJECT_ROOT``.

    - Project root: ``myfile.csv``
    - Kaggle drop folder: ``data/kaggle/myfile.csv``
    - Archive (training splits, etc.): ``archive/myfile.csv``
    """
    items: list[str] = []
    DATA_KAGGLE_DIR.mkdir(parents=True, exist_ok=True)
    for p in sorted(PROJECT_ROOT.glob("*.csv")):
        items.append(p.name)
    if DATA_KAGGLE_DIR.is_dir():
        for p in sorted(DATA_KAGGLE_DIR.glob("*.csv")):
            rel = p.relative_to(PROJECT_ROOT).as_posix()
            items.append(rel)
    if ARCHIVE_DIR.is_dir():
        for p in sorted(ARCHIVE_DIR.glob("*.csv")):
            rel = p.relative_to(PROJECT_ROOT).as_posix()
            items.append(rel)
    return sorted(set(items), key=lambda s: s.lower())


def default_csv_name(filenames: list[str]) -> str:
    """Prefer ``Stocks.csv`` (any location), then ``stock_tweets.csv``, else first path."""
    if not filenames:
        return ""
    for f in filenames:
        if Path(f).name == PRIORITY_DEFAULT_CSV:
            return f
    for f in filenames:
        if Path(f).name == "stock_tweets.csv":
            return f
    return filenames[0]


def resolve_dataset_path(data_path: str | Path) -> Path:
    """
    Resolve a dropdown value (e.g. ``data/kaggle/foo.csv`` or ``foo.csv``) to an absolute path.
    """
    raw = Path(data_path)
    if raw.is_file():
        return raw.resolve()
    candidate = (PROJECT_ROOT / data_path).resolve()
    if candidate.is_file():
        return candidate
    by_name = (PROJECT_ROOT / raw.name).resolve()
    if by_name.is_file():
        return by_name
    return candidate


def find_column(columns: list[str], candidates: list[str]) -> Optional[str]:
    """Exact match first; substring only for tokens length >= 5 (avoids post → GPA_Post_AI)."""
    lowered = {str(c).lower().strip(): c for c in columns}
    for candidate in candidates:
        key = candidate.lower().strip()
        if key in lowered:
            return lowered[key]
    for candidate in candidates:
        key = candidate.lower().strip()
        if len(key) < 5:
            continue
        for column_name, original in lowered.items():
            if key in column_name:
                return original
    return None


def find_date_column(columns: list[str]) -> Optional[str]:
    lowered = {c.lower().strip(): c for c in columns}
    for key in ("date", "timestamp", "datetime", "created_at", "submitted_at"):
        if key in lowered:
            return lowered[key]
    for column_name, original in lowered.items():
        for token in ("created_at", "timestamp", "datetime"):
            if token in column_name and "time_saved" not in column_name:
                return original
    return None


def _object_columns_excluding_ids(data: pd.DataFrame) -> list[str]:
    cols: list[str] = []
    for column in data.columns:
        label = str(column).strip()
        if ID_LIKE_COLUMN.match(label):
            continue
        if COMBINE_COLUMN_BLOCK.search(label):
            continue
        if data[column].dtype == object or pd.api.types.is_string_dtype(data[column]):
            sample = data[column].dropna().astype(str).head(200)
            if len(sample) == 0:
                continue
            if sample.str.match(NUMERIC_STRING).mean() > 0.85:
                continue
            if sample.str.len().mean() < 2.5:
                continue
            cols.append(column)
    return cols


def _combine_text_columns(data: pd.DataFrame, columns: list[str]) -> Optional[pd.Series]:
    if not columns:
        return None
    acc = data[columns[0]].fillna("").astype(str)
    for column in columns[1:]:
        acc = acc + " " + data[column].fillna("").astype(str)
    return acc.str.replace(r"\s+", " ", regex=True).str.strip()


def load_dataset_from_path(data_path: str | Path) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    Read CSV and map to standard columns.

    Returns (dataframe, error_message). On success error_message is None.
    """
    path = resolve_dataset_path(data_path)
    if not path.is_file():
        return None, f"Dataset file not found: {data_path}"

    data = pd.read_csv(path)
    if data.empty:
        return None, "Dataset is empty."

    cols = list(data.columns)
    tweet_col = find_column(cols, TEXT_COLUMN_CANDIDATES)
    obj_cols = _object_columns_excluding_ids(data)

    if tweet_col:
        text_series = data[tweet_col].astype(str)
    elif obj_cols:
        text_series = _combine_text_columns(data, obj_cols)
        if text_series is None or text_series.str.len().mean() < 2:
            return None, "No usable text column found in this CSV."
    else:
        return None, "No text or string columns found in this CSV."

    date_col = find_date_column(cols)
    ticker_col = find_column(cols, ["stock name", "ticker", "symbol", "brand", "product"])
    company_col = find_column(cols, ["company", "company name", "major", "department", "category"])

    if date_col:
        data[date_col] = pd.to_datetime(data[date_col], errors="coerce")

    dataset = pd.DataFrame(
        {
            "date": data[date_col] if date_col else pd.NaT,
            "tweet": text_series.astype(str),
            "ticker": data[ticker_col].astype(str) if ticker_col else "",
            "company": data[company_col].astype(str) if company_col else "",
        }
    )
    return dataset, None


def apply_date_filters(filtered: pd.DataFrame, start_date: Optional[str], end_date: Optional[str]) -> pd.DataFrame:
    """Keep NaT dates when a range is set so survey-style CSVs are not emptied."""
    if not start_date and not end_date:
        return filtered
    out = filtered
    if start_date:
        ts = pd.to_datetime(start_date, utc=True, errors="coerce")
        out = out[out["date"].isna() | (out["date"] >= ts)]
    if end_date:
        ts = pd.to_datetime(end_date, utc=True, errors="coerce")
        out = out[out["date"].isna() | (out["date"] <= ts)]
    return out


def create_get_enriched_cached(enrich_fn):
    """
    Return a cached (path -> enriched DataFrame) loader.

    enrich_fn: Callable[[pd.DataFrame], pd.DataFrame] — e.g. partial with SIA bound.
    """

    @lru_cache(maxsize=16)
    def get_enriched_csv(path_resolved_str: str) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
        dataset, err = load_dataset_from_path(path_resolved_str)
        if err:
            return None, err
        return enrich_fn(dataset), None

    return get_enriched_csv
