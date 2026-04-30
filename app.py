from collections import Counter
from pathlib import Path
import re

import dash
import dash_bootstrap_components as dbc
from dash import dash_table, dcc, html
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from textblob import TextBlob

DATA_PATHS = [
    Path(__file__).resolve().with_name("stock_tweets.csv"),
]
MA_WINDOWS = (7, 14, 30)
SENTIMENT_ORDER = ["Positive", "Neutral", "Negative"]
TOKEN_PATTERN = re.compile(r"[A-Za-z]{2,}")


def _find_column(columns, candidates):
    for candidate in candidates:
        for column in columns:
            lowered = column.lower()
            if lowered == candidate:
                return column
            if candidate in lowered:
                return column
    return None


def _format_percent(value):
    if value is None or pd.isna(value):
        return "N/A"
    return f"{value * 100:.2f}%"


def _format_number(value):
    if value is None or pd.isna(value):
        return "N/A"
    return f"{value:,.2f}"


def ensure_nltk_data():
    resources = [
        ("sentiment/vader_lexicon.zip", "vader_lexicon"),
        ("tokenizers/punkt.zip", "punkt"),
    ]
    for resource_path, resource_name in resources:
        try:
            nltk.data.find(resource_path)
        except LookupError:
            nltk.download(resource_name)


def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r"@[A-Za-z0-9_]+", "", text)
    text = re.sub(r"#", "", text)
    text = re.sub(r"RT[\s]+", "", text)
    text = re.sub(r"https?://\S+", "", text)
    return text.strip()


def get_final_sentiment(combined_score):
    if combined_score >= 0.05:
        return "Positive"
    if combined_score <= -0.05:
        return "Negative"
    return "Neutral"


def get_data_path():
    for path in DATA_PATHS:
        if path.exists():
            return path
    return DATA_PATHS[0]


def load_data():
    data_path = get_data_path()
    if not data_path.exists():
        return None, None, data_path

    df = pd.read_csv(data_path)
    if df.empty:
        return df, None, data_path

    columns = list(df.columns)
    date_col = _find_column(columns, [
        "date",
        "timestamp",
        "datetime",
        "created_at",
        "createdat",
    ])
    if date_col:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.sort_values(date_col)

    return df, date_col, data_path


def analyze_sentiment(df):
    tweet_col = _find_column(
        list(df.columns),
        ["tweet", "text", "full_text"],
    )
    if not tweet_col:
        return df, None

    ensure_nltk_data()
    df = df.copy()
    df["raw_tweet"] = df[tweet_col].astype(str)
    df["cleaned_tweet"] = df["raw_tweet"].apply(clean_text)

    df["textblob_polarity"] = df["cleaned_tweet"].apply(
        lambda text: TextBlob(text).sentiment.polarity
    )
    df["textblob_subjectivity"] = df["cleaned_tweet"].apply(
        lambda text: TextBlob(text).sentiment.subjectivity
    )

    sia = SentimentIntensityAnalyzer()
    df["vader_compound"] = df["cleaned_tweet"].apply(
        lambda text: sia.polarity_scores(text)["compound"]
    )
    df["combined_sentiment_score"] = (
        df["textblob_polarity"] + df["vader_compound"]
    ) / 2
    df["final_sentiment"] = df["combined_sentiment_score"].apply(get_final_sentiment)
    return df, tweet_col


def build_sentiment_chart(df):
    if df.empty or "final_sentiment" not in df.columns:
        return px.bar(title="No sentiment data available")

    sentiment_counts = (
        df["final_sentiment"]
        .value_counts()
        .reindex(SENTIMENT_ORDER)
        .fillna(0)
        .astype(int)
    )
    chart_df = sentiment_counts.reset_index()
    chart_df.columns = ["sentiment", "count"]

    fig = px.bar(
        chart_df,
        x="sentiment",
        y="count",
        color="sentiment",
        color_discrete_map={"Positive": "#2E8B57", "Neutral": "#6C757D", "Negative": "#C0392B"},
        title="Sentiment Distribution",
        text="count",
    )
    fig.update_layout(margin=dict(l=20, r=20, t=40, b=20), xaxis_title=None, yaxis_title="Tweets")
    return fig


def build_polarity_histogram(df):
    if df.empty or "combined_sentiment_score" not in df.columns:
        return px.histogram(title="No sentiment data available")
    fig = px.histogram(
        df,
        x="combined_sentiment_score",
        nbins=30,
        title="Combined Sentiment Score",
        color_discrete_sequence=["#1F77B4"],
    )
    fig.update_layout(margin=dict(l=20, r=20, t=40, b=20), xaxis_title="Score", yaxis_title="Tweets")
    return fig


def build_scatter(df):
    if df.empty or "textblob_polarity" not in df.columns:
        return px.scatter(title="No sentiment data available")
    fig = px.scatter(
        df,
        x="textblob_polarity",
        y="textblob_subjectivity",
        color="final_sentiment",
        color_discrete_map={"Positive": "#2E8B57", "Neutral": "#6C757D", "Negative": "#C0392B"},
        hover_data=["cleaned_tweet"],
        title="Polarity vs Subjectivity",
    )
    fig.update_layout(margin=dict(l=20, r=20, t=40, b=20), xaxis_title="Polarity", yaxis_title="Subjectivity")
    return fig


def build_top_terms(df, top_n=12):
    if df.empty or "cleaned_tweet" not in df.columns:
        return px.bar(title="No terms available")
    counter = Counter()
    for text in df["cleaned_tweet"].dropna().astype(str):
        tokens = [token.lower() for token in TOKEN_PATTERN.findall(text)]
        counter.update(tokens)
    most_common = counter.most_common(top_n)
    if not most_common:
        return px.bar(title="No terms available")
    terms_df = pd.DataFrame(most_common, columns=["term", "count"])
    fig = px.bar(
        terms_df,
        x="count",
        y="term",
        orientation="h",
        title="Top Terms",
        color_discrete_sequence=["#FF8C42"],
    )
    fig.update_layout(margin=dict(l=20, r=20, t=40, b=20), xaxis_title="Count", yaxis_title=None)
    return fig


def build_sentiment_timeline(df, date_col):
    if df.empty or not date_col or "combined_sentiment_score" not in df.columns:
        return px.line(title="No sentiment time series available")
    timeline = (
        df.dropna(subset=[date_col])
        .groupby(pd.Grouper(key=date_col, freq="D"))["combined_sentiment_score"]
        .mean()
        .reset_index()
    )
    fig = px.line(
        timeline,
        x=date_col,
        y="combined_sentiment_score",
        title="Daily Average Sentiment",
        markers=True,
    )
    fig.update_layout(margin=dict(l=20, r=20, t=40, b=20), yaxis_title="Sentiment")
    return fig


def build_price_figure(df, date_col, price_col, sentiment_col):
    fig = make_subplots(specs=[[{"secondary_y": sentiment_col is not None}]])
    x_values = df[date_col] if date_col else df.index

    fig.add_trace(
        go.Scatter(x=x_values, y=df[price_col], name=price_col, line=dict(width=2)),
        secondary_y=False,
    )

    for window in MA_WINDOWS:
        ma_col = f"MA{window}"
        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=df[ma_col],
                name=ma_col,
                line=dict(width=1, dash="dot"),
            ),
            secondary_y=False,
        )

    if sentiment_col:
        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=df[sentiment_col],
                name=sentiment_col,
                line=dict(width=2, color="#FF6F3C"),
            ),
            secondary_y=True,
        )
        fig.update_yaxes(title_text="Sentiment", secondary_y=True)

    fig.update_layout(
        template="plotly_white",
        height=320,
        margin=dict(l=20, r=20, t=40, b=20),
        title="Price with Moving Averages",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_yaxes(title_text="Price", secondary_y=False)
    return fig


def build_volume_figure(df, date_col, volume_col):
    fig = go.Figure()
    x_values = df[date_col] if date_col else df.index
    fig.add_trace(
        go.Bar(x=x_values, y=df[volume_col], name=volume_col, marker_color="#4C78A8")
    )
    fig.update_layout(
        template="plotly_white",
        height=320,
        margin=dict(l=20, r=20, t=40, b=20),
        title="Volume",
    )
    return fig


def build_returns_figure(df, date_col, returns_col):
    fig = go.Figure()
    x_values = df[date_col] if date_col else df.index
    fig.add_trace(
        go.Scatter(x=x_values, y=df[returns_col], name="Returns", line=dict(width=1.5))
    )
    fig.update_layout(
        template="plotly_white",
        height=260,
        margin=dict(l=20, r=20, t=40, b=20),
        title="Returns",
    )
    return fig


def build_tweet_activity_figure(daily_df):
    if daily_df.empty:
        return px.line(title="No activity data available")
    fig = px.line(
        daily_df,
        x="date",
        y="tweet_count",
        title="Tweet Activity (Daily)",
        markers=True,
    )
    fig.update_layout(margin=dict(l=20, r=20, t=40, b=20), yaxis_title="Tweets")
    return fig


def build_tweet_volume_figure(daily_df):
    if daily_df.empty:
        return px.bar(title="No activity data available")
    fig = px.bar(
        daily_df,
        x="date",
        y="tweet_count",
        title="Tweet Volume (Daily)",
        color_discrete_sequence=["#4C78A8"],
    )
    fig.update_layout(margin=dict(l=20, r=20, t=40, b=20), yaxis_title="Tweets")
    return fig


def build_sentiment_change_figure(daily_df):
    if daily_df.empty or "sentiment_change" not in daily_df.columns:
        return px.line(title="No sentiment change data available")
    fig = px.line(
        daily_df,
        x="date",
        y="sentiment_change",
        title="Sentiment Change (Daily)",
        markers=True,
    )
    fig.update_layout(margin=dict(l=20, r=20, t=40, b=20), yaxis_title="Change")
    return fig


def placeholder_figure(title, message, height=320):
    fig = go.Figure()
    fig.add_annotation(text=message, x=0.5, y=0.5, showarrow=False)
    fig.update_layout(
        template="plotly_white",
        height=height,
        margin=dict(l=20, r=20, t=40, b=20),
        title=title,
    )
    return fig


def build_ticker_comparison(df, ticker_col):
    if df.empty or not ticker_col or "combined_sentiment_score" not in df.columns:
        return px.bar(title="No ticker comparison data available"), pd.DataFrame()

    grouped = (
        df.groupby(ticker_col)
        .agg(
            tweets=(ticker_col, "size"),
            avg_sentiment=("combined_sentiment_score", "mean"),
            positive_share=(
                "final_sentiment",
                lambda series: (series == "Positive").mean() * 100,
            ),
            negative_share=(
                "final_sentiment",
                lambda series: (series == "Negative").mean() * 100,
            ),
        )
        .reset_index()
        .sort_values("tweets", ascending=False)
    )
    top_two = grouped.head(2)

    fig = px.bar(
        top_two,
        x=ticker_col,
        y="avg_sentiment",
        color=ticker_col,
        title="Top Tickers: Avg Sentiment",
        text=top_two["avg_sentiment"].map(lambda v: f"{v:.2f}"),
    )
    fig.update_layout(margin=dict(l=20, r=20, t=40, b=20), xaxis_title=None, yaxis_title="Sentiment")
    table = top_two.rename(columns={ticker_col: "ticker"})
    return fig, table


def build_tweet_tables(df, date_col):
    if df.empty or "combined_sentiment_score" not in df.columns:
        empty = pd.DataFrame(columns=["cleaned_tweet", "combined_sentiment_score", "final_sentiment"])
        return empty, empty, empty
    positive = df.sort_values("combined_sentiment_score", ascending=False).head(10)
    negative = df.sort_values("combined_sentiment_score", ascending=True).head(10)
    if date_col and date_col in df.columns:
        recent = df.sort_values(date_col, ascending=False).head(15)
    else:
        recent = df.tail(15)
    return positive, negative, recent


def make_table(df, max_rows=10):
    display_df = df.copy()
    if display_df.empty:
        display_df = pd.DataFrame(columns=df.columns)
    return dash_table.DataTable(
        data=display_df.to_dict("records"),
        columns=[{"name": col.replace("_", " ").title(), "id": col} for col in display_df.columns],
        page_size=max_rows,
        style_cell={"textAlign": "left", "padding": "8px", "whiteSpace": "normal", "height": "auto"},
        style_header={"backgroundColor": "#F0F3F7", "fontWeight": "bold"},
        style_table={"overflowX": "auto"},
    )


df, date_col, active_path = load_data()

price_fig = placeholder_figure("Price with Moving Averages", "No price column found")
volume_fig = placeholder_figure("Volume", "No volume column found")
returns_fig = placeholder_figure("Returns", "No returns data", height=260)
sentiment_fig = placeholder_figure("Sentiment Distribution", "No tweet column found")
polarity_fig = placeholder_figure("Combined Sentiment Score", "No tweet column found")
scatter_fig = placeholder_figure("Polarity vs Subjectivity", "No tweet column found")
top_terms_fig = placeholder_figure("Top Terms", "No tweet column found", height=360)
timeline_fig = placeholder_figure("Daily Average Sentiment", "No tweet column found")
ticker_fig = placeholder_figure("Top Tickers", "No ticker data")

ticker_value = "N/A"
sentiment_latest = None
last_return = None
volatility = None
total_tweets = 0
avg_polarity = None
avg_subjectivity = None
positive_share = None
negative_share = None
top_ticker_table = pd.DataFrame(columns=["ticker", "tweets", "avg_sentiment", "positive_share", "negative_share"])
pos_table = pd.DataFrame(columns=["cleaned_tweet", "combined_sentiment_score"])
neg_table = pd.DataFrame(columns=["cleaned_tweet", "combined_sentiment_score"])
recent_table = pd.DataFrame(columns=["cleaned_tweet", "final_sentiment", "combined_sentiment_score"])

if df is not None and not df.empty:
    columns = list(df.columns)
    price_col = _find_column(columns, [
        "adj_close",
        "close",
        "last",
        "price",
    ])
    volume_col = _find_column(columns, ["volume", "vol"])
    ticker_col = _find_column(columns, ["ticker", "symbol", "stock name", "stock", "company"])

    if ticker_col:
        ticker_series = df[ticker_col].dropna()
        if not ticker_series.empty:
            ticker_value = str(ticker_series.mode().iloc[0])

    df, tweet_col = analyze_sentiment(df)
    if tweet_col:
        sentiment_latest = df["combined_sentiment_score"].dropna().iloc[-1] if not df.empty else None
        total_tweets = len(df)
        avg_polarity = df["textblob_polarity"].mean()
        avg_subjectivity = df["textblob_subjectivity"].mean()
        positive_share = (df["final_sentiment"] == "Positive").mean()
        negative_share = (df["final_sentiment"] == "Negative").mean()

        sentiment_fig = build_sentiment_chart(df)
        polarity_fig = build_polarity_histogram(df)
        scatter_fig = build_scatter(df)
        top_terms_fig = build_top_terms(df)
        timeline_fig = build_sentiment_timeline(df, date_col)

        if ticker_col:
            ticker_fig, top_ticker_table = build_ticker_comparison(df, ticker_col)

        pos_table, neg_table, recent_table = build_tweet_tables(df, date_col)

    daily_df = pd.DataFrame()
    if date_col:
        daily_df = (
            df.dropna(subset=[date_col])
            .groupby(pd.Grouper(key=date_col, freq="D"))
            .agg(
                tweet_count=(ticker_col if ticker_col else df.columns[0], "size"),
                avg_sentiment=("combined_sentiment_score", "mean"),
            )
            .reset_index()
        )
        daily_df = daily_df.rename(columns={date_col: "date"})
        daily_df["sentiment_change"] = daily_df["avg_sentiment"].diff()

    if price_col:
        df[price_col] = pd.to_numeric(df[price_col], errors="coerce")
        for window in MA_WINDOWS:
            df[f"MA{window}"] = df[price_col].rolling(window=window).mean()

        returns_col = "returns"
        df[returns_col] = df[price_col].pct_change()
        returns_clean = df[returns_col].dropna()
        if not returns_clean.empty:
            last_return = returns_clean.iloc[-1]
            volatility = returns_clean.tail(30).std()

        price_fig = build_price_figure(df, date_col, price_col, "combined_sentiment_score")
        returns_fig = build_returns_figure(df, date_col, returns_col)
    elif not daily_df.empty:
        price_fig = build_tweet_activity_figure(daily_df)
        returns_fig = build_sentiment_change_figure(daily_df)

    if volume_col:
        df[volume_col] = pd.to_numeric(df[volume_col], errors="coerce")
        volume_fig = build_volume_figure(df, date_col, volume_col)
    elif not daily_df.empty:
        volume_fig = build_tweet_volume_figure(daily_df)


app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)
server = app.server

app.index_string = """
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>Twitter Stock Sentiment Dashboard</title>
        {%favicon%}
        {%css%}
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
        <link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&display=swap" rel="stylesheet">
        <style>
            :root {
                --ink: #0F172A;
                --muted: #64748B;
                --card: #FFFFFF;
                --bg: #F1F5F9;
                --accent: #2563EB;
                --accent-soft: #DBEAFE;
                --success: #16A34A;
                --danger: #DC2626;
            }
            * { box-sizing: border-box; }
            body {
                font-family: "Space Grotesk", sans-serif;
                background: radial-gradient(circle at 10% 10%, #E0F2FE 0%, #F8FAFC 45%, #EEF2FF 100%);
                color: var(--ink);
            }
            .app-shell { padding: 18px 16px 32px; }
            .sidebar {
                background: var(--card);
                border-radius: 18px;
                padding: 20px;
                box-shadow: 0 16px 40px rgba(15, 23, 42, 0.08);
                min-height: 100%;
            }
            .nav-pill {
                background: var(--accent-soft);
                color: var(--ink);
                border-radius: 999px;
                padding: 6px 14px;
                font-weight: 600;
                font-size: 0.85rem;
                display: inline-block;
            }
            .kpi-card {
                border: none;
                border-radius: 16px;
                box-shadow: 0 12px 28px rgba(15, 23, 42, 0.08);
                background: var(--card);
            }
            .kpi-value { font-size: 1.6rem; font-weight: 700; }
            .kpi-label { color: var(--muted); font-size: 0.85rem; }
            .panel-card {
                background: var(--card);
                border-radius: 18px;
                padding: 16px;
                box-shadow: 0 14px 32px rgba(15, 23, 42, 0.08);
            }
            .tab-pane { padding-top: 12px; }
            .dash-table-container .dash-spreadsheet-container { font-family: "Space Grotesk", sans-serif; }
            @media (max-width: 992px) {
                .app-shell { padding: 12px; }
                .sidebar { margin-bottom: 16px; }
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
"""

app.layout = dbc.Container(
    html.Div(
        [
            dbc.Row(
                [
                    dbc.Col(
                        html.Div(
                            [
                                html.Div("Twitter Stock Sentiment", className="nav-pill mb-3"),
                                html.H2("Portfolio Sentiment Dashboard", className="fw-bold"),
                                html.P(
                                    "Market mood, tweet-driven signals, and stock comparisons.",
                                    className="text-muted",
                                ),
                                html.Div("Overview", className="fw-semibold mt-4"),
                                html.Div("Compare", className="text-muted"),
                                html.Div("Tweets", className="text-muted"),
                                html.Hr(),
                                html.Div("Primary Ticker", className="text-muted"),
                                html.H4(ticker_value, className="fw-bold"),
                                html.Div(
                                    f"Source: {(active_path.name if active_path else 'stock_tweets.csv')}",
                                    className="text-muted mt-3",
                                ),
                            ],
                            className="sidebar",
                        ),
                        md=3,
                    ),
                    dbc.Col(
                        html.Div(
                            [
                                dbc.Row(
                                    [
                                        dbc.Col(
                                            dbc.Card(
                                                dbc.CardBody(
                                                    [
                                                        html.Div("Total Tweets", className="kpi-label"),
                                                        html.Div(total_tweets, className="kpi-value"),
                                                    ]
                                                ),
                                                className="kpi-card",
                                            ),
                                            md=3,
                                        ),
                                        dbc.Col(
                                            dbc.Card(
                                                dbc.CardBody(
                                                    [
                                                        html.Div("Avg Sentiment", className="kpi-label"),
                                                        html.Div(_format_number(sentiment_latest), className="kpi-value"),
                                                    ]
                                                ),
                                                className="kpi-card",
                                            ),
                                            md=3,
                                        ),
                                        dbc.Col(
                                            dbc.Card(
                                                dbc.CardBody(
                                                    [
                                                        html.Div("Positive Share", className="kpi-label"),
                                                        html.Div(_format_percent(positive_share), className="kpi-value"),
                                                    ]
                                                ),
                                                className="kpi-card",
                                            ),
                                            md=3,
                                        ),
                                        dbc.Col(
                                            dbc.Card(
                                                dbc.CardBody(
                                                    [
                                                        html.Div("Negative Share", className="kpi-label"),
                                                        html.Div(_format_percent(negative_share), className="kpi-value"),
                                                    ]
                                                ),
                                                className="kpi-card",
                                            ),
                                            md=3,
                                        ),
                                    ],
                                    className="g-3",
                                ),
                                dbc.Tabs(
                                    [
                                        dbc.Tab(
                                            html.Div(
                                                [
                                                    dbc.Row(
                                                        [
                                                            dbc.Col(dcc.Graph(figure=timeline_fig), md=8),
                                                            dbc.Col(dcc.Graph(figure=sentiment_fig), md=4),
                                                        ],
                                                        className="g-3 mt-2",
                                                    ),
                                                    dbc.Row(
                                                        [
                                                            dbc.Col(dcc.Graph(figure=polarity_fig), md=4),
                                                            dbc.Col(dcc.Graph(figure=scatter_fig), md=4),
                                                            dbc.Col(dcc.Graph(figure=top_terms_fig), md=4),
                                                        ],
                                                        className="g-3 mt-2",
                                                    ),
                                                    dbc.Row(
                                                        [
                                                            dbc.Col(
                                                                html.Div(
                                                                    [
                                                                        html.H5("Recent Tweets", className="fw-semibold"),
                                                                        make_table(
                                                                            recent_table[[
                                                                                "cleaned_tweet",
                                                                                "final_sentiment",
                                                                                "combined_sentiment_score",
                                                                            ]],
                                                                            max_rows=10,
                                                                        ),
                                                                    ],
                                                                    className="panel-card",
                                                                ),
                                                                md=12,
                                                            ),
                                                        ],
                                                        className="g-3 mt-2",
                                                    ),
                                                ]
                                            ),
                                            label="Overview",
                                            tabClassName="ms-1",
                                        ),
                                        dbc.Tab(
                                            html.Div(
                                                [
                                                    dbc.Row(
                                                        [
                                                            dbc.Col(dcc.Graph(figure=price_fig), md=7),
                                                            dbc.Col(dcc.Graph(figure=volume_fig), md=5),
                                                        ],
                                                        className="g-3 mt-2",
                                                    ),
                                                    dbc.Row(
                                                        [
                                                            dbc.Col(dcc.Graph(figure=returns_fig), md=7),
                                                            dbc.Col(dcc.Graph(figure=ticker_fig), md=5),
                                                        ],
                                                        className="g-3 mt-2",
                                                    ),
                                                    dbc.Row(
                                                        [
                                                            dbc.Col(
                                                                html.Div(
                                                                    [
                                                                        html.H5("Top Ticker Comparison", className="fw-semibold"),
                                                                        make_table(top_ticker_table, max_rows=5),
                                                                    ],
                                                                    className="panel-card",
                                                                ),
                                                                md=12,
                                                            ),
                                                        ],
                                                        className="g-3 mt-2",
                                                    ),
                                                ]
                                            ),
                                            label="Compare",
                                        ),
                                        dbc.Tab(
                                            html.Div(
                                                [
                                                    dbc.Row(
                                                        [
                                                            dbc.Col(
                                                                html.Div(
                                                                    [
                                                                        html.H5("Top Positive Tweets", className="fw-semibold"),
                                                                        make_table(
                                                                            pos_table[["cleaned_tweet", "combined_sentiment_score"]],
                                                                            max_rows=8,
                                                                        ),
                                                                    ],
                                                                    className="panel-card",
                                                                ),
                                                                md=6,
                                                            ),
                                                            dbc.Col(
                                                                html.Div(
                                                                    [
                                                                        html.H5("Top Negative Tweets", className="fw-semibold"),
                                                                        make_table(
                                                                            neg_table[["cleaned_tweet", "combined_sentiment_score"]],
                                                                            max_rows=8,
                                                                        ),
                                                                    ],
                                                                    className="panel-card",
                                                                ),
                                                                md=6,
                                                            ),
                                                        ],
                                                        className="g-3 mt-2",
                                                    ),
                                                    dbc.Row(
                                                        [
                                                            dbc.Col(
                                                                html.Div(
                                                                    [
                                                                        html.H5("Sentiment Coverage", className="fw-semibold"),
                                                                        html.P(
                                                                            "These tables summarize sentiment extremes, similar to the fees breakdown in the financial dashboard.",
                                                                            className="text-muted",
                                                                        ),
                                                                    ],
                                                                    className="panel-card",
                                                                ),
                                                                md=12,
                                                            ),
                                                        ],
                                                        className="g-3 mt-2",
                                                    ),
                                                ]
                                            ),
                                            label="Tweets",
                                        ),
                                    ],
                                    className="mt-3",
                                ),
                            ]
                        ),
                        md=9,
                    ),
                ],
                className="g-3",
            )
        ],
        className="app-shell",
    ),
    fluid=True,
)


if __name__ == "__main__":
    app.run(debug=True)
