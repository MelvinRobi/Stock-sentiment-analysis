"""Dash layout: minimalist black/white shell."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from dash import dash_table, dcc, html
import dash_bootstrap_components as dbc

INDEX_CSS = """
html, body { background: #000000 !important; color: #FFFFFF !important; }
.text-muted, .text-secondary { color: #E5E5E5 !important; }
h1, h2, h3, h4, h5, h6, p, span, div, label { color: inherit; }
.card {
    background-color: #0A0A0A !important;
    border: 1px solid #2A2A2A !important;
    color: #FFFFFF !important;
}
.card-body { color: #FFFFFF !important; }
.form-control {
    background-color: #000000 !important;
    color: #FFFFFF !important;
    border: 1px solid #444444 !important;
}
.Select-control, .Select-value-label, .Select-input input {
    background-color: #000000 !important;
    color: #FFFFFF !important;
    border-color: #444444 !important;
}
.VirtualizedSelectOption { color: #FFFFFF !important; background-color: #111111 !important; }
.Select-menu-outer { background-color: #111111 !important; border: 1px solid #444444 !important; }
.dash-spreadsheet-container .dash-spreadsheet-inner td,
.dash-spreadsheet-container .dash-spreadsheet-inner th { color: #FFFFFF !important; }
.DateInput_input, .DateRangePickerInput {
    background: #000000 !important;
    color: #FFFFFF !important;
    border-color: #444444 !important;
}
.btn-primary { background: #FFFFFF !important; color: #000000 !important; border: 1px solid #FFFFFF !important; }
.btn-outline-light:hover { background: #FFFFFF !important; color: #000000 !important; }
.btn-outline-light { border-color: #888888 !important; color: #FFFFFF !important; }
a, .rc-slider-mark-text { color: #FFFFFF !important; }
.section-title { font-size: 0.7rem; letter-spacing: 0.12em; text-transform: uppercase; color: #AAAAAA; }
"""


def build_index_string(title: str = "Sentiment analysis") -> str:
    return f"""
<!DOCTYPE html>
<html>
    <head>
        {{%metas%}}
        <title>{title}</title>
        {{%favicon%}}
        {{%css%}}
        <style>{INDEX_CSS}</style>
    </head>
    <body>
        {{%app_entry%}}
        <footer>{{%config%}}{{%scripts%}}{{%renderer%}}</footer>
    </body>
</html>
"""


def build_table(df, page_size: int = 12):
    """Monochrome DataTable for message lists."""
    display = df.copy()
    if display.empty:
        display = pd.DataFrame(columns=getattr(df, "columns", ["Tweet", "Score"]))
    return dash_table.DataTable(
        data=display.to_dict("records"),
        columns=[{"name": c, "id": c} for c in display.columns],
        page_size=page_size,
        style_cell={
            "textAlign": "left",
            "padding": "10px",
            "whiteSpace": "normal",
            "height": "auto",
            "backgroundColor": "#000000",
            "color": "#FFFFFF",
            "border": "1px solid #333333",
            "fontFamily": "system-ui, sans-serif",
            "fontSize": "0.9rem",
        },
        style_header={
            "fontWeight": "600",
            "backgroundColor": "#111111",
            "color": "#FFFFFF",
            "border": "1px solid #333333",
        },
        style_table={"overflowX": "auto", "border": "1px solid #333333"},
    )


def csv_dropdown_options(csv_filenames: list[str]) -> list[dict]:
    """Labels for the dataset dropdown (root vs Kaggle folder)."""
    out = []
    for f in csv_filenames:
        norm = f.replace("\\", "/")
        if norm.startswith("data/kaggle/"):
            label = f"Kaggle · {Path(f).name}"
        else:
            label = Path(f).name
        out.append({"label": label, "value": f})
    return out


def build_layout(csv_filenames: list[str], default_csv: str) -> dbc.Container:
    csv_options = csv_dropdown_options(csv_filenames)
    return dbc.Container(
        [
            dcc.Store(id="message-tab-store", data="Positive"),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.Div("Sentiment analysis", className="section-title mb-1"),
                            html.H2("Dashboard", className="fw-bold mb-0"),
                            html.P(
                                "TextBlob + NLTK VADER · local CSV or StockTwits",
                                className="text-secondary small mb-0",
                            ),
                        ],
                        md=12,
                    ),
                ],
                className="pt-4 pb-3",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    html.Div("Data source", className="section-title mb-2"),
                                    dcc.RadioItems(
                                        id="source-select",
                                        options=[
                                            {"label": "CSV file", "value": "csv"},
                                            {"label": "StockTwits (live)", "value": "stocktwits"},
                                        ],
                                        value="csv",
                                        inline=True,
                                        className="d-flex flex-wrap gap-3",
                                        labelStyle={
                                            "color": "#FFFFFF",
                                            "fontWeight": "500",
                                            "cursor": "pointer",
                                            "marginRight": "1rem",
                                        },
                                        inputStyle={"cursor": "pointer"},
                                    ),
                                ]
                            )
                        ),
                        md=12,
                    ),
                ],
                className="mb-3",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.Div("Dataset", className="section-title mb-1"),
                            dcc.Dropdown(
                                id="csv-select",
                                options=csv_options,
                                value=default_csv or None,
                                clearable=False,
                            ),
                        ],
                        md=12,
                        lg=5,
                    ),
                    dbc.Col(
                        [
                            html.Div("Keyword", className="section-title mb-1"),
                            dbc.Input(
                                id="product-input",
                                type="text",
                                placeholder="Optional for CSV — leave empty for capped sample",
                                value="",
                            ),
                        ],
                        md=12,
                        lg=4,
                    ),
                    dbc.Col(
                        dbc.Button("Run analysis", id="analyze-btn", color="primary", className="w-100 mt-4"),
                        md=6,
                        lg=1,
                    ),
                    dbc.Col(
                        dbc.Button("Refresh live", id="refresh-btn", outline=True, color="light", className="w-100 mt-4"),
                        md=6,
                        lg=2,
                    ),
                ],
                className="g-2 mb-3",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        dcc.DatePickerRange(
                            id="date-range",
                            display_format="YYYY-MM-DD",
                            clearable=True,
                        ),
                        md=12,
                        lg=6,
                    ),
                    dbc.Col(
                        [
                            html.Div("Minimum rows", className="section-title mb-0"),
                            dcc.Slider(
                                id="min-matches",
                                min=1,
                                max=300,
                                step=1,
                                value=5,
                                marks={1: "1", 50: "50", 150: "150", 300: "300"},
                            ),
                        ],
                        md=12,
                        lg=6,
                    ),
                ],
                className="mb-3",
            ),
            dbc.Alert(id="status-message", color="dark", className="py-2", style={"border": "1px solid #333"}),
            dbc.Row(
                [
                    dbc.Col(dbc.Card(dbc.CardBody([html.Div("Rows", className="section-title"), html.H4(id="kpi-count", className="mb-0")])), md=3),
                    dbc.Col(dbc.Card(dbc.CardBody([html.Div("Avg score", className="section-title"), html.H4(id="kpi-sentiment", className="mb-0")])), md=3),
                    dbc.Col(dbc.Card(dbc.CardBody([html.Div("Positive %", className="section-title"), html.H4(id="kpi-positive", className="mb-0")])), md=3),
                    dbc.Col(dbc.Card(dbc.CardBody([html.Div("Neutral %", className="section-title"), html.H4(id="kpi-neutral", className="mb-0")])), md=3),
                ],
                className="g-2 mb-3",
            ),
            dcc.Loading(
                type="circle",
                color="#FFFFFF",
                children=dbc.Row(
                    [
                        dbc.Col(dcc.Graph(id="distribution-graph"), lg=6),
                        dbc.Col(dcc.Graph(id="timeline-graph"), lg=6),
                    ],
                    className="g-3",
                ),
            ),
            dcc.Loading(
                type="circle",
                color="#FFFFFF",
                children=dbc.Row(
                    [
                        dbc.Col(dcc.Graph(id="terms-graph"), lg=6),
                        dbc.Col(dcc.Graph(id="breakdown-graph"), lg=6),
                    ],
                    className="g-3 mt-1",
                ),
            ),
            html.Div("Messages by sentiment", className="section-title mt-4 mb-2"),
            dbc.ButtonGroup(
                [
                    dbc.Button("Positive", id="msg-tab-positive", outline=True, color="light", size="sm"),
                    dbc.Button("Neutral", id="msg-tab-neutral", outline=True, color="light", size="sm"),
                    dbc.Button("Negative", id="msg-tab-negative", outline=True, color="light", size="sm"),
                ],
                className="mb-2",
            ),
            html.Div(id="messages-table", className="mb-5"),
        ],
        fluid=True,
        style={"minHeight": "100vh", "paddingBottom": "32px", "maxWidth": "1400px"},
    )
