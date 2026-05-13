"""
Initial data exploration for archive train/test CSVs.

Run as module::

    python -m sentiment_training.exploration
    python -m sentiment_training.exploration --train-nrows 50000
"""

from __future__ import annotations

import argparse
import json
from typing import Any, Dict, Optional

import pandas as pd

from sentiment_training.io import load_train_test
from sentiment_training.labels import NAME_BY_ID


def summarize_split(name: str, df: pd.DataFrame) -> Dict[str, Any]:
    """Structured summary for logging or JSON export."""
    counts = df["sentiment"].value_counts().sort_index()
    counts_named = {NAME_BY_ID.get(int(k), str(k)): int(v) for k, v in counts.items()}
    return {
        "split": name,
        "n_rows": int(len(df)),
        "columns": list(df.columns),
        "dtypes": {c: str(df[c].dtype) for c in df.columns},
        "sentiment_counts": counts.to_dict(),
        "sentiment_counts_named": counts_named,
        "head_rows": df.head(5).assign(
            label_name=lambda x: x["sentiment"].map(NAME_BY_ID)
        ).to_dict(orient="records"),
    }


def format_report(summaries: Dict[str, Dict[str, Any]]) -> str:
    """Human-readable multi-split report."""
    lines = []
    for name, s in summaries.items():
        lines.append(f"=== {s['split']} ({s['n_rows']:,} rows) ===")
        lines.append(f"Columns: {s['columns']}")
        lines.append(f"dtypes: {s['dtypes']}")
        lines.append(f"Label counts (numeric): {s['sentiment_counts']}")
        lines.append(f"Label counts (named): {s['sentiment_counts_named']}")
        lines.append("First 5 rows (with label_name):")
        for row in s["head_rows"]:
            snippet = str(row.get("sentence", ""))[:120].replace("\n", " ")
            lines.append(
                f"  sentiment={row.get('sentiment')} ({row.get('label_name')}) | {snippet}..."
            )
        lines.append("")
    lines.append(
        "Note: This archive uses binary labels only (0=negative, 1=positive). "
        "There is no neutral class in train_data.csv / test_data.csv."
    )
    return "\n".join(lines)


def run_exploration(train_nrows: Optional[int], export_json: Optional[str] = None) -> Dict[str, Any]:
    train, test = load_train_test(train_nrows=train_nrows, validate_labels=True)
    # Attach names only for head display in summarize (already in summarize via assign)
    summaries = {
        "train": summarize_split("train", train),
        "test": summarize_split("test", test),
    }
    print(format_report(summaries))
    if export_json:
        path = export_json
        with open(path, "w", encoding="utf-8") as f:
            json.dump(summaries, f, indent=2)
        print(f"Wrote JSON summary to {path}")
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(description="Explore archive train/test CSVs")
    parser.add_argument(
        "--train-nrows",
        type=int,
        default=10_000,
        help="Rows to read from train (default 10000). Omit with a very large number for full train.",
    )
    parser.add_argument("--full-train", action="store_true", help="Load entire train file (slow, large memory).")
    parser.add_argument("--json-out", type=str, default=None, help="Optional path to write JSON summary.")
    args = parser.parse_args()
    nrows = None if args.full_train else args.train_nrows
    run_exploration(train_nrows=nrows, export_json=args.json_out)


if __name__ == "__main__":
    main()
