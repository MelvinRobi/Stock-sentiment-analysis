# Kaggle CSV drop folder

Put any CSV you download from **Kaggle** (or elsewhere) **directly in this folder**.

Example path on disk:

```
data/kaggle/sentiment_training.csv
```

Then:

1. Restart the Dash app (`python3 app.py`) so the new file appears in the **Dataset** dropdown as **Kaggle · sentiment_training.csv**.
2. Choose **CSV file** as the data source.
3. Click **Run analysis** (optional keyword to filter rows).

The loader maps common text column names automatically; see `sentiment_dashboard/data_loader.py` if you need to extend patterns.
