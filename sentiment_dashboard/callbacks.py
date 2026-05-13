"""Dash callback wiring: CSV + StockTwits, chart updates, sentiment message buckets.

Restart the app to clear the in-memory CSV parse cache after editing a file on disk.
"""

from __future__ import annotations

from functools import partial

import dash
import pandas as pd
from dash import Input, Output, State

from sentiment_dashboard import charts
from sentiment_dashboard.config import MAX_CSV_ROWS_NO_KEYWORD
from sentiment_dashboard.data_loader import (
    apply_date_filters,
    create_get_enriched_cached,
    default_csv_name,
    discover_csv_filenames,
    resolve_dataset_path,
)
from sentiment_dashboard.live_feeds import fetch_stocktwits_messages
from sentiment_dashboard.sentiment import enrich_sentiment
from sentiment_dashboard.ui import build_table

MESSAGE_ROWS = 120


def _default_response(message: str):
    empty = build_table(pd.DataFrame(columns=["Tweet", "Score"]))
    return (
        message,
        "0",
        "N/A",
        "N/A",
        "N/A",
        charts.make_empty_figure("Distribution", "—"),
        charts.make_empty_figure("Timeline", "—"),
        charts.make_empty_figure("Terms", "—"),
        charts.make_empty_figure("Breakdown", "—"),
        empty,
        "Positive",
    )


def _message_slice(filtered: pd.DataFrame, sentiment: str) -> pd.DataFrame:
    sub = filtered[filtered["sentiment"] == sentiment][["clean_tweet", "score"]].copy()
    sub = sub.rename(columns={"clean_tweet": "Tweet", "score": "Score"})
    if not sub.empty:
        sub["Score"] = sub["Score"].round(3)
    return sub.head(MESSAGE_ROWS)


def register_callbacks(app, sia) -> None:
    get_enriched_csv = create_get_enriched_cached(partial(enrich_sentiment, sia=sia))

    @app.callback(
        [
            Output("status-message", "children"),
            Output("kpi-count", "children"),
            Output("kpi-sentiment", "children"),
            Output("kpi-positive", "children"),
            Output("kpi-neutral", "children"),
            Output("distribution-graph", "figure"),
            Output("timeline-graph", "figure"),
            Output("terms-graph", "figure"),
            Output("breakdown-graph", "figure"),
            Output("messages-table", "children"),
            Output("message-tab-store", "data"),
        ],
        [
            Input("analyze-btn", "n_clicks"),
            Input("refresh-btn", "n_clicks"),
            Input("csv-select", "value"),
            Input("source-select", "value"),
            Input("msg-tab-positive", "n_clicks"),
            Input("msg-tab-neutral", "n_clicks"),
            Input("msg-tab-negative", "n_clicks"),
        ],
        [
            State("product-input", "value"),
            State("date-range", "start_date"),
            State("date-range", "end_date"),
            State("min-matches", "value"),
            State("message-tab-store", "data"),
        ],
        prevent_initial_call=False,
    )
    def on_analyze(
        _a,
        _r,
        csv_select_value,
        source,
        _mp,
        _mn,
        _mz,
        product_query,
        start_date,
        end_date,
        min_matches,
        prev_tab,
    ):
        ctx = dash.callback_context
        if not ctx.triggered:
            triggered = ""
        else:
            triggered = ctx.triggered[0]["prop_id"].split(".")[0]

        tab_map = {
            "msg-tab-positive": "Positive",
            "msg-tab-neutral": "Neutral",
            "msg-tab-negative": "Negative",
        }

        if triggered in tab_map:
            message_tab = tab_map[triggered]
        elif triggered in ("analyze-btn", "refresh-btn", "csv-select", "source-select"):
            message_tab = "Positive"
        else:
            message_tab = prev_tab if prev_tab in ("Positive", "Neutral", "Negative") else "Positive"

        csv_names = discover_csv_filenames()
        default_name = default_csv_name(csv_names)
        csv_name = (csv_select_value or default_name or "").strip()
        query = (product_query or "").strip()

        if not csv_name and source == "csv":
            return (*_default_response("No CSV files in project folder."), "Positive")

        filtered: pd.DataFrame | None = None
        source_note = ""

        if source == "csv":
            csv_path = resolve_dataset_path(csv_name).resolve()
            base_df, csv_err = get_enriched_csv(str(csv_path))
            if csv_err or base_df is None:
                return (*_default_response(csv_err or "Could not load CSV."), message_tab)
            if not query:
                filtered = base_df.copy()
                note = ""
                if len(filtered) > MAX_CSV_ROWS_NO_KEYWORD:
                    filtered = filtered.head(MAX_CSV_ROWS_NO_KEYWORD).copy()
                    note = f" (first {MAX_CSV_ROWS_NO_KEYWORD} rows; add keyword to narrow)"
                source_note = f"CSV · {csv_name}{note}"
            else:
                mask = (
                    base_df["tweet"].str.lower().str.contains(query.lower(), na=False)
                    | base_df["ticker"].str.lower().str.contains(query.lower(), na=False)
                    | base_df["company"].str.lower().str.contains(query.lower(), na=False)
                )
                filtered = base_df[mask].copy()
                source_note = f"CSV · {csv_name}"
        else:
            if not query:
                return (
                    *_default_response("Enter a ticker or company name for StockTwits."),
                    message_tab,
                )
            live_df, err = fetch_stocktwits_messages(query)
            if err:
                return (*_default_response(err), message_tab)
            filtered = enrich_sentiment(live_df)
            source_note = "StockTwits"

        filtered = apply_date_filters(filtered, start_date, end_date)

        if filtered.empty:
            hint = f"No rows for keyword '{query}'." if query else "No rows after filters."
            return (*_default_response(f"{hint} Clear dates or lower minimum rows."), message_tab)

        if len(filtered) < (min_matches or 1):
            return (
                *_default_response(
                    f"Only {len(filtered)} row(s). Lower “Minimum rows” or widen filters."
                ),
                message_tab,
            )

        chart_label = query if query else "sample"
        dist, line, terms, br = charts.build_charts(filtered, chart_label)
        total = len(filtered)
        avg_s = filtered["score"].mean()
        pos_pct = (filtered["sentiment"] == "Positive").mean() * 100
        neu_pct = (filtered["sentiment"] == "Neutral").mean() * 100
        latest = filtered["date"].dropna().max()
        latest_lbl = latest.strftime("%Y-%m-%d %H:%M UTC") if pd.notna(latest) else "—"

        msg_df = _message_slice(filtered, message_tab)
        status = f"{source_note} · {total} rows · messages: {message_tab} · last date: {latest_lbl}"

        return (
            status,
            str(total),
            f"{avg_s:.3f}",
            f"{pos_pct:.1f}%",
            f"{neu_pct:.1f}%",
            dist,
            line,
            terms,
            br,
            build_table(msg_df, page_size=15),
            message_tab,
        )
