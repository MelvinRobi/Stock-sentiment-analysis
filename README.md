# CT Sentiment Analyzer

A comprehensive sentiment analysis dashboard built with **Dash** that analyzes sentiment in CSV datasets using multiple NLP techniques. The application combines **TextBlob**, **NLTK VADER**, and **Transformer-based** models for accurate sentiment classification, with interactive visualizations and real-time data exploration capabilities.

## Overview

CT Sentiment Analyzer is a production-ready web application designed to:
- Analyze sentiment across large text datasets with multiple algorithms
- Visualize sentiment trends, distributions, and patterns
- Compare sentiment scores from different NLP models
- Support custom CSV uploads from local files or Kaggle datasets
- Provide detailed metrics and moving averages for temporal analysis

## Features

### Core Functionality
- **Multi-Model Sentiment Analysis**: Combines TextBlob polarity, NLTK VADER intensity scores, and transformer-based sentiment models
- **Flexible Data Import**: Load CSV files from local storage or dedicated Kaggle folder
- **Interactive Dashboard**: Real-time filtering, searching, and chart interactions using Plotly
- **Sentiment Visualization**: 
  - Distribution charts (positive, neutral, negative)
  - Time-series trends with moving averages (7-day, 14-day, 30-day)
  - Keyword-specific sentiment breakdown
  - Comparative sentiment analysis
- **Performance Optimization**: Caching and efficient data loading for datasets up to 8,000+ rows
- **Dark Theme UI**: Modern, professional black/white interface for extended viewing

### Sentiment Models
1. **TextBlob**: Fast polarity analysis (-1 to 1 scale)
2. **NLTK VADER**: Valence Aware Dictionary and sEntiment Reasoner, optimized for social media
3. **Transformer Models**: Hugging Face transformers for state-of-the-art accuracy

## Project Structure

```
app.py                           # Main Dash application entrypoint
requirements.txt                 # Python dependencies and versions
plotly-cloud.toml               # Plotly cloud configuration
Procfile                        # Deployment configuration for Heroku/cloud

sentiment_dashboard/            # Dashboard module
├── ui.py                       # Dash layout and UI components
├── callbacks.py                # Interactive callbacks and logic
├── config.py                   # Configuration constants
├── data_loader.py              # CSV loading and data discovery
└── __init__.py

sentiment_training/             # ML training and preprocessing utilities
├── __init__.py
├── config.py                   # Training configuration
├── exploration.py              # Data exploration utilities
├── io.py                       # File I/O operations
├── labels.py                   # Label management
├── pipeline.py                 # Training pipeline
├── preprocess.py               # Text preprocessing
└── README.md

transformer_sentiment.py         # Transformer model inference wrapper
notebooks/
├── Stock_sentiment.ipynb       # Analysis and exploration notebook

archive/                        # Archived data and legacy files
├── README.md
├── exploring_data.ipynb        # Original data exploration
├── train_data.csv              # Training dataset
├── test_data.csv               # Test dataset
├── vocab.py                    # Vocabulary utilities
└── vocab.json                  # Vocabulary mappings

data/
└── kaggle/                     # Custom Kaggle datasets (empty by default)

docs/
└── KAGGLE_DATASET.md           # Kaggle data import guide

notebooks/                      # Jupyter notebooks for analysis
└── Stock_sentiment.ipynb
```

## Installation & Setup

### Prerequisites
- Python 3.8+
- pip (Python package manager)
- Virtual environment support

### Quick Start

1. **Clone/Navigate to the project**:
   ```bash
   cd /path/to/Stock-sentiment-analysis
   ```

2. **Create and activate virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   This installs:
   - `dash` & `dash-bootstrap-components`: Web framework
   - `pandas` & `numpy`: Data manipulation
   - `plotly`: Interactive visualizations
   - `nltk`: VADER sentiment analysis
   - `textblob`: TextBlob polarity scores
   - `transformers` & `torch`: Transformer models
   - `gunicorn`: Production server

4. **Run the application**:
   ```bash
   python app.py
   ```

5. **Access the dashboard**:
   - Open **http://127.0.0.1:8050** in your browser
   - The app will auto-detect available CSV files and load a default dataset

## Usage Guide

### Loading Data

#### Using Built-in CSV Files
- `stock_tweets.csv`: Pre-loaded tweets about stocks
- `Stocks.csv`: Stock-related data (if present)
- Any root-level CSV file

#### Adding Kaggle Datasets

1. **Download from Kaggle**:
   - Visit [Kaggle Datasets](https://www.kaggle.com/datasets)
   - Download any CSV file (e.g., product reviews, tweets, comments)

2. **Import into the app**:
   - Copy the CSV file to `data/kaggle/` directory
   - Restart the application: `python app.py`
   - In the dashboard: **Data source → CSV file**
   - **Dataset dropdown → Kaggle · [filename]**
   - Click **Run Analysis**

3. **Default loading**:
   - The app prioritizes `Stocks.csv` if present
   - Falls back to `stock_tweets.csv`
   - Then loads the first file alphabetically
   - You can override via the UI dropdown

#### Custom Column Detection
The app auto-detects text columns for analysis:
- **Primary text column**: Searches for "text", "tweet", "comment", "content", "message"
- **Combine mode**: Merges multiple text columns if primary not found
- **Edit detection**: Modify `sentiment_dashboard/data_loader.py`:
  ```python
  TEXT_COLUMN_CANDIDATES = ["text", "content", "tweet", "description"]
  COMBINE_COLUMN_BLOCK = ["id", "user", "timestamp"]  # columns to skip
  ```

### Dashboard Features

#### Analysis Controls
- **Data Source**: Choose between CSV file or real-time StockTwits stream
- **Dataset Selector**: Pick which CSV to analyze
- **Keyword Filter**: Search for specific terms in the text
- **Sentiment Model**: Toggle between TextBlob, VADER, or Transformer
- **Run Analysis**: Process and visualize the selected data

#### Visualizations
- **Sentiment Distribution**: Pie chart showing positive/neutral/negative breakdown
- **Time Series**: Line chart with configurable moving averages (7/14/30 days)
- **Top Keywords**: Bar chart of most common words by sentiment
- **Data Table**: Searchable, paginated view of all records with scores

#### Performance Metrics
- Total records analyzed
- Average sentiment score
- Distribution percentages
- Processing time

## Advanced Configuration

### Training Pipeline (Archive Data)

The `sentiment_training/` module supports classifier training:

```bash
# Explore training data
python -m sentiment_training.exploration --train-nrows 50000

# Run full pipeline
python -m sentiment_training.pipeline
```

See `archive/README.md` for detailed training documentation.

### Transformer Models

Edit `transformer_sentiment.py` to:
- Change the model: `model_name = "distilbert-base-uncased-finetuned-sst-2-english"`
- Adjust batch size for memory constraints
- Enable GPU acceleration with `device='cuda'`

### Column Customization

Edit `sentiment_dashboard/data_loader.py`:

```python
# Primary text column search patterns
TEXT_COLUMN_CANDIDATES = [
    "text", "tweet", "comment", "content", "message", "body"
]

# Columns to exclude when auto-combining
COMBINE_COLUMN_BLOCK = [
    "id", "timestamp", "user_id", "username", "date"
]
```

## API & Callbacks

### Sentiment Scoring
```python
from transformer_sentiment import score_texts

scores = score_texts(texts=['I love this!', 'Terrible experience'])
# Returns: [{'positive': 0.99, 'negative': 0.01}, ...]
```

### Data Loading
```python
from sentiment_dashboard.data_loader import discover_csv_filenames, resolve_dataset_path

files = discover_csv_filenames()  # Discover all available CSVs
path = resolve_dataset_path("data/kaggle/my_data.csv")  # Resolve file path
```

## Deployment

### Heroku Deployment

1. **Setup Heroku CLI** and authenticate
2. **Create Procfile** (included):
   ```
   web: gunicorn app:server
   ```
3. **Deploy**:
   ```bash
   git push heroku main
   ```

### Docker Deployment

Create `Dockerfile`:
```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["gunicorn", "app:server"]
```

Build and run:
```bash
docker build -t ct-sentiment .
docker run -p 8050:8050 ct-sentiment
```

## Troubleshooting

### CSV Not Appearing in Dropdown
- Ensure file is in root directory, `data/kaggle/`, or `archive/`
- Restart the application
- Check file is valid CSV with readable columns

### Sentiment Scores All Zero/Same
- Verify text column detection in data_loader.py
- Check CSV has actual text data (not empty/null)
- Run `python app.py` with verbose logging

### Performance Issues with Large Files
- Limit rows: Modify `MAX_ROWS_NO_KEYWORD = 8000` in app.py
- Use keyword filter to reduce dataset size
- Switch to TextBlob (faster) instead of Transformer model

### Missing NLTK Data
- The app auto-downloads VADER lexicon on first run
- If error persists: `python -m nltk.downloader vader_lexicon`

## Dependencies

See `requirements.txt` for full list. Key packages:
- `dash==2.x`: Web framework
- `plotly==5.x`: Visualization
- `pandas>=1.3`: Data manipulation
- `nltk>=3.6`: VADER sentiment
- `textblob>=0.17`: TextBlob sentiment
- `transformers>=4.20`: Transformer models
- `torch>=1.10`: Deep learning (CPU/GPU)

## Contributing

To extend the sentiment analysis:

1. **Add new sentiment models**: Implement in `transformer_sentiment.py`
2. **Custom visualizations**: Add charts in `sentiment_dashboard/ui.py`
3. **Preprocessing improvements**: Enhance `sentiment_training/preprocess.py`
4. **Testing**: Add tests in dedicated `tests/` directory

## License

This project is provided as-is for educational and commercial use.

## Support

For issues, questions, or feature requests, please refer to the documentation in `docs/` or review the code comments.

After **replacing** a CSV on disk with the same path, restart the app to clear the parse cache.
