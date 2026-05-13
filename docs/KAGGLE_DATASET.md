# Kaggle workflow (short)

1. On [Kaggle](https://www.kaggle.com), open a dataset → **Download** the CSV (or export from a notebook).
2. Copy the file into **`data/kaggle/`** in this project (see `data/kaggle/README.md`).
3. Restart **`python3 app.py`**, select the file under **Dataset**, run analysis.

Column mapping is automatic; extend `TEXT_COLUMN_CANDIDATES` in `sentiment_dashboard/data_loader.py` if needed.
