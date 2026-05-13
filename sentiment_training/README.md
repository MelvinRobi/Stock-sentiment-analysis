# Sentiment training utilities

Modular helpers for the **`archive/`** CSVs (and future datasets with the same schema).

## Layout

| Module | Role |
|--------|------|
| `config.py` | Paths to `train_data.csv`, `test_data.csv` |
| `labels.py` | Binary label schema and string names |
| `io.py` | `load_train()`, `load_test()`, `load_train_test()` |
| `exploration.py` | Summaries: shape, dtypes, head, label validation |
| `preprocess.py` | Text cleaning + word-level tokenization |

**Labels:** The bundled `train_data.csv` / `test_data.csv` use **binary** integers `0` (negative) and `1` (positive) only — not a three-way positive/neutral/negative schema. The first rows of the training file are often all one class; use a larger `train_nrows` or `python -m sentiment_training.exploration --full-train` for full label counts (slow).

## Quick start

```python
from sentiment_training.io import load_train_test
from sentiment_training.exploration import summarize_split
from sentiment_training.preprocess import clean_text, tokenize

train, test = load_train_test(train_nrows=10_000)  # omit for full train (slow)
print(summarize_split("train", train))
tokens = tokenize(clean_text(train.loc[0, "sentence"]))
```

CLI exploration report:

```bash
python -m sentiment_training.exploration
```

Optional: limit train rows for speed:

```bash
python -m sentiment_training.exploration --train-nrows 50000
```

## Next steps (model training)

1. Apply `clean_text` / `tokenize` (or batch via `apply_clean_tokenize`).
2. Vectorize (TF-IDF, embeddings) or feed token ids from your tokenizer.
3. Train classifier; use `test_data.csv` only for final evaluation.
