"""Application paths and constants. Adjust here when adding new datasets."""

from pathlib import Path

# Project root (parent of this package)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Drop Kaggle (or any) exports here — see data/kaggle/README.md
DATA_KAGGLE_DIR = PROJECT_ROOT / "data" / "kaggle"
# Labeled train/test splits and related artifacts — see archive/README.md
ARCHIVE_DIR = PROJECT_ROOT / "archive"
MAX_CSV_ROWS_NO_KEYWORD = 8000

# Preferred default CSV if present (e.g. stock export). Falls back to first *.csv
PRIORITY_DEFAULT_CSV = "Stocks.csv"

# StockTwits symbol stream
STOCKTWITS_SYMBOL_URL = "https://api.stocktwits.com/api/2/streams/symbol/{symbol}.json"
STOCKTWITS_DEFAULT_LIMIT = 30

# Plotly: subtle motion on updates (ms)
CHART_TRANSITION_MS = 450
