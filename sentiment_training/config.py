"""Paths relative to project root (directory containing app.py)."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARCHIVE_DIR = PROJECT_ROOT / "archive"

TRAIN_CSV = ARCHIVE_DIR / "train_data.csv"
TEST_CSV = ARCHIVE_DIR / "test_data.csv"

# Expected columns after load
REQUIRED_COLUMNS = ("sentence", "sentiment")
