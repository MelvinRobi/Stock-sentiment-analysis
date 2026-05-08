# Stock Sentiment Analysis Dashboard

A comprehensive web application that combines financial market data with Twitter sentiment analysis to provide real-time insights into stock performance and public opinion trends.

## 🚀 Features

### **Sentiment Analysis**
- **Dual Engine**: Combines TextBlob and VADER sentiment analysis
- **Real-time Processing**: Analyzes tweet sentiment on-the-fly
- **Classification**: Automatic categorization into Positive/Neutral/Negative
- **Scoring**: Combined sentiment scores with confidence metrics

### **Financial Analysis**
- **Price Tracking**: Stock prices with 7/14/30-day moving averages
- **Volume Analysis**: Trading volume patterns and trends
- **Returns Calculation**: Daily returns and volatility metrics
- **Correlation Analysis**: Sentiment vs. price movement relationships

### **Interactive Visualizations**
- **Sentiment Distribution**: Real-time sentiment category breakdowns
- **Timeline Analysis**: Daily sentiment trends and patterns
- **Price Charts**: Interactive stock charts with sentiment overlay
- **Activity Metrics**: Tweet volume and engagement tracking
- **Top Terms**: Most frequently mentioned terms in tweets
- **Scatter Analysis**: Polarity vs. subjectivity relationships

### **Data Intelligence**
- **Multi-ticker Support**: Compare sentiment across different stocks
- **Tweet Tables**: Top positive, negative, and recent tweets
- **Auto-detection**: Smart column detection for various data formats
- **Time Series Processing**: Automatic date normalization and sorting

## 📋 Project Structure

```
Stock-sentiment-analysis/
├── app.py                    # Main Dash application (859 lines)
├── requirements.txt          # Python dependencies
├── stock_tweets.csv         # Dataset (~18MB)
├── Procfile                 # Heroku deployment config
├── plotly-cloud.toml        # Plotly cloud deployment
├── README.md               # This file
├── notebooks/
│   └── Stock_sentiment.ipynb  # Analysis notebook (~15MB)
└── venv/                   # Virtual environment
```

## 🛠️ Technology Stack

- **Backend**: Python 3.x
- **Web Framework**: Dash with Plotly
- **UI Components**: Dash Bootstrap Components
- **Data Processing**: Pandas, NumPy
- **Sentiment Analysis**: NLTK (VADER), TextBlob
- **Machine Learning**: Scikit-learn
- **Visualization**: Plotly, Plotly Express
- **Deployment**: Gunicorn, Heroku/Plotly Cloud

## 📊 Data Requirements

The application expects a CSV file (`stock_tweets.csv`) with the following structure:

### **Required Columns (auto-detected)**
- **Date**: `date`, `timestamp`, `datetime`, `created_at`, `createdat`
- **Tweet Text**: `tweet`, `text`, `full_text`
- **Stock Symbol**: `ticker`, `symbol`, `stock name`, `stock`, `company`
- **Price**: `adj_close`, `close`, `last`, `price`
- **Volume**: `volume`, `vol`

### **Sample Data Format**
```csv
date,tweet,ticker,close,volume
2023-01-01,"Tesla stock is amazing! $TSLA",TSLA,150.25,1000000
2023-01-01,"Bad day for Apple investors",AAPL,125.50,800000
```

## 🚀 Quick Start

### **1. Environment Setup**
```bash
# Clone the repository
git clone <repository-url>
cd Stock-sentiment-analysis

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```

### **2. Install Dependencies**
```bash
pip install -r requirements.txt
```

### **3. Prepare Data**
- Place your `stock_tweets.csv` file in the project root
- The app will auto-detect column names
- First run will download required NLTK data automatically

### **4. Run the Application**
```bash
python app.py
```

### **5. Access the Dashboard**
Open your browser and navigate to: **http://127.0.0.1:8050**

## 📖 Usage Guide

### **Dashboard Navigation**
1. **Main Dashboard**: Overview with key metrics and charts
2. **Sentiment Analysis**: Detailed sentiment breakdowns and trends
3. **Financial Charts**: Stock prices, volume, and returns
4. **Tweet Analysis**: Top tweets and term frequency analysis
5. **Comparison Tools**: Multi-ticker sentiment comparison

### **Key Metrics Displayed**
- **Latest Sentiment**: Current sentiment score
- **Tweet Volume**: Total number of tweets analyzed
- **Average Polarity**: Overall sentiment polarity
- **Positive/Negative Ratio**: Sentiment distribution percentages
- **Stock Performance**: Price changes and volatility
- **Correlation Metrics**: Sentiment vs. price relationships

### **Interactive Features**
- **Date Range Selection**: Filter data by time period
- **Ticker Selection**: Analyze specific stocks
- **Sentiment Filters**: Focus on positive/negative tweets
- **Export Functionality**: Download charts and data
- **Real-time Updates**: Dynamic chart updates

## 🔧 Configuration

### **Customizing Column Detection**
Edit the candidate lists in `app.py` if your CSV uses different column names:

```python
# Example: Custom date column detection
date_col = _find_column(columns, [
    "date",
    "timestamp", 
    "datetime",
    "created_at",
    "createdat",
    "your_custom_date_column",  # Add your column name here
])
```

### **Sentiment Thresholds**
Adjust sentiment classification thresholds:

```python
def get_final_sentiment(combined_score):
    if combined_score >= 0.05:    # Positive threshold
        return "Positive"
    if combined_score <= -0.05:   # Negative threshold
        return "Negative"
    return "Neutral"
```

### **Moving Average Windows**
Modify MA calculation periods:

```python
MA_WINDOWS = (7, 14, 30)  # 7-day, 14-day, 30-day moving averages
```

## 📓 Analysis Notebook

### **Exploratory Data Analysis**
Open `notebooks/Stock_sentiment.ipynb` for:
- **Data Profiling**: Comprehensive dataset analysis
- **Sentiment Deep Dive**: Detailed sentiment analysis
- **Visualization Gallery**: Extended chart collection
- **Statistical Analysis**: Correlations and trends
- **Model Testing**: Sentiment model performance

### **Notebook Contents**
- **Data Loading and Preprocessing**
- **Sentiment Analysis Pipeline**
- **Financial Metrics Calculation**
- **Advanced Visualizations**
- **Statistical Tests and Insights**

## 🚀 Deployment

### **Heroku Deployment**
```bash
# Install Heroku CLI
heroku login
heroku create your-app-name
git push heroku main
```

### **Plotly Cloud Deployment**
The `plotly-cloud.toml` file contains deployment configuration for Plotly's cloud platform.

### **Docker Deployment** (Optional)
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["gunicorn", "app:server"]
```

## 🔍 Troubleshooting

### **Common Issues**

**1. NLTK Data Missing**
```bash
# Download manually if auto-download fails
python -c "import nltk; nltk.download('vader_lexicon'); nltk.download('punkt')"
```

**2. Column Detection Fails**
- Verify your CSV has the expected columns
- Check for exact column name matches
- Add custom column names to the candidate lists

**3. Memory Issues with Large Datasets**
- Filter data by date range
- Use data sampling for initial testing
- Consider using Dask for very large datasets

**4. Port Already in Use**
```bash
# Run on different port
python app.py --port 8051
```

### **Performance Optimization**
- Use data sampling for development
- Implement caching for repeated analyses
- Consider database integration for production
- Optimize sentiment analysis with parallel processing

## 📈 API Integration

### **Twitter API Integration**
For real-time data, integrate with Twitter API:
```python
# Example Twitter API integration
import tweepy
# Add your API keys and streaming logic
```

### **Stock Market APIs**
Enhance with real-time market data:
- Alpha Vantage
- Yahoo Finance
- IEX Cloud
- Polygon.io

## 🤝 Contributing

### **Development Setup**
```bash
# Clone and setup development environment
git clone <repository-url>
cd Stock-sentiment-analysis
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
pip install -r requirements-dev.txt  # if available
```

### **Code Style**
- Follow PEP 8 guidelines
- Use type hints where applicable
- Add docstrings to functions
- Include unit tests for new features

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📞 Support

For questions, issues, or contributions:
- Create an issue on GitHub
- Check the troubleshooting section
- Review the analysis notebook for examples

---

**Note**: This application is designed for educational and research purposes. Financial decisions should not be based solely on sentiment analysis from social media.
