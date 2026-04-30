# Twitter Stock Analysis

Dash app and notebook for exploring stock price and sentiment signals from tweets.

## Quick Start

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the app:

```bash
python app.py
```

Open http://127.0.0.1:8050 in your browser.

## Data

The dashboard reads the CSV from `stock_tweets.csv` in the project root. If it is not
found, it falls back to `D:\Projects\school\Data-analysis\stock_tweets.csv`. It attempts
to detect common column names for dates, prices, volume, and sentiment. If the file
contains tweet text, the app computes TextBlob and VADER sentiment metrics.

## Notebook

Open `notebooks/data_exploration.ipynb` to load the CSV, preview rows, and inspect
the inferred schema.

## Notes

- The app includes price moving averages, volume, returns, and optional sentiment overlay.
- If your CSV uses different column names, adjust the candidates in `app.py`.
- The app will download NLTK data on first run if it is missing.
