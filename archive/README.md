# Archive — training assets

Structured layout for labeled sentiment CSVs and tokenizer artifacts.

| File | Purpose |
|------|---------|
| `train_data.csv` | Large training split (`sentence`, `sentiment`) |
| `test_data.csv` | Held-out evaluation split (same columns) |
| `exploring_data.ipynb` | Notebook for ad-hoc exploration |
| `vocab.py` | Builds a BPE tokenizer from text (requires `tokenizers`) |
| `vocab.json` | Serialized tokenizer produced by `vocab.py` |

## Label encoding (this dataset)

Both CSVs use **numeric binary** labels only:

| Value | Meaning |
|-------|---------|
| `0` | Negative |
| `1` | Positive |

There is **no neutral class** in these files. For three-way (positive / neutral / negative) modeling, use a different dataset or derive neutral via thresholds later.

Row counts (approx.): train ~1.52M lines including header; test ~359 rows.

## Python pipeline

Use the **`sentiment_training`** package at the repo root for loading, exploration summaries, and text preprocessing — see `sentiment_training/README.md`.
