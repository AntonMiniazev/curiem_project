import base64
import math
import os
from dataclasses import dataclass
from datetime import date, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

API_BASE = os.getenv("API_BASE_URL", "https://api.ampere-data.work")
DEFAULT_DAYS = 30
COLOR_SEQUENCE = [
    ("Deep Space Blue", "#02040f"),
    ("Prussian Blue", "#002642"),
    ("Space Indigo", "#211d3e"),
    ("Midnight Violet", "#42133a"),
    ("Crimson Violet", "#630a36"),
    ("Dark Amaranth", "#840032"),
    ("Vintage Berry", "#9d375c"),
    ("Blush Rose", "#b56d86"),
    ("Peach Nectar", "#f5b19f"),
    ("Pink Orchid", "#cda4b0"),
]
AVG_LINE_COLOR = "#EB134B"
CURRENCY_SYMBOL = "₣"

# Must be the first Streamlit command.
st.set_page_config(page_title="Ampere Retail Performance", layout="wide")


def format_currency(value: Optional[float], decimal: int = 2) -> str:
    if value is None or (
        isinstance(value, (int, float)) and math.isclose(value, 0.0, abs_tol=1e-9)
    ):
        return "-"
    abs_val = abs(value)
    formatted = f"{abs_val:,.{decimal}f}"
    if value < 0:
        return f"{CURRENCY_SYMBOL} ({formatted})"
    return f"{CURRENCY_SYMBOL} {formatted}"


def format_percentage(value: Optional[float]) -> str:
    if value is None or (
        isinstance(value, (int, float)) and math.isclose(value, 0.0, abs_tol=1e-9)
    ):
        return "-"
    formatted = f"{abs(value):.2f}%"
    return f"({formatted})" if value < 0 else formatted


def build_hover_data(values: List[float]) -> List[List[str]]:
    return [[format_currency(val)] for val in values]


def _read_env_api_key() -> Optional[str]:
    """Load API_KEY from a local .env file if present."""
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if not env_path.exists():
        return None
    for line in env_path.read_text().splitlines():
        if "=" not in line or line.strip().startswith("#"):
            continue
        key, _, value = line.partition("=")
        if key.strip() == "API_KEY":
            return value.strip()
    return None


def _get_api_key() -> str:
    """Prefer Streamlit secrets, then environment variables, then .env file."""
    secret_key: Optional[str] = None
    if hasattr(st, "secrets"):
        try:
            secret_key = st.secrets["API_KEY"]
        except (FileNotFoundError, KeyError):
            secret_key = None
    if secret_key:
        return secret_key
    env_key = os.getenv("API_KEY")
    if env_key:
        return env_key
    file_key = _read_env_api_key()
    if file_key:
        return file_key
    raise RuntimeError(
        "API_KEY is not configured in Streamlit secrets, environment, or .env"
    )


@lru_cache(maxsize=1)
def _resolved_api_key() -> str:
    """Lazy-resolve API key after Streamlit page config is set."""
    return _get_api_key()


@dataclass
class Filters:
    start_date: date
    end_date: date
    store_name: Optional[str]

    def to_params(self) -> Dict[str, str]:
        params = {
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat(),
        }
        if self.store_name:
            params["store_name"] = self.store_name
        return params


def _auth_headers() -> Dict[str, str]:
    return {"X-API-Key": _resolved_api_key()}


def _safe_request(
    path: str, params: Optional[Dict[str, Any]] = None, retries: int = 2
) -> Dict[str, Any]:
    url = f"{API_BASE}{path}"
    last_error: Optional[Exception] = None
    for attempt in range(retries + 1):
        try:
            response = requests.get(
                url,
                headers=_auth_headers(),
                params=params,
                timeout=20,
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as exc:
            last_error = exc
    raise last_error or RuntimeError("Unknown API error")


@st.cache_data(show_spinner=False)
def fetch_json(path: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return _safe_request(path, params)


@st.cache_data(show_spinner=False)
def load_filter_options() -> Dict[str, List[str]]:
    data = fetch_json("/filters/options")
    return {
        "stores": data.get("stores", []),
    }


def render_filters() -> Filters:
    st.sidebar.markdown("### Filters")
    end_default = date.today()
    start_default = date(end_default.year, end_default.month, 1)
    options = load_filter_options()

    start = st.sidebar.date_input("Start date", value=start_default)
    end = st.sidebar.date_input("End date", value=end_default)
    if start > end:
        st.sidebar.error("Start date must be before end date")

    store_choice = st.sidebar.selectbox(
        "Store", ["All stores"] + options["stores"], index=0
    )

    store_name = None if store_choice == "All stores" else store_choice

    return Filters(start_date=start, end_date=end, store_name=store_name)


def render_metrics(filters: Filters):
    try:
        data = fetch_json("/metrics/summary", filters.to_params())
    except Exception as exc:
        st.error(f"Unable to load summary metrics: {exc}")
        return
    summary = data.get("summary", {})
    sales = summary.get("sales", 0.0)
    gross_profit = summary.get("gross_profit", 0.0)
    total_orders = summary.get("total_orders", 0)
    avg_order_value = (sales / total_orders) if total_orders else None
    gp_percent = (gross_profit / sales * 100) if sales else None

    cols = st.columns(5)
    cols[0].metric("Average order value", format_currency(avg_order_value, 0))
    cols[1].metric("Sales", format_currency(sales, 0))
    cols[2].metric(
        "Cost of sales", format_currency(summary.get("cost_of_sales", 0.0), 0)
    )
    cols[3].metric("Gross profit", format_currency(gross_profit, 0))
    cols[4].metric("Gross profit, %", format_percentage(gp_percent))


def render_sales_trend(filters: Filters):
    daily_view = st.toggle("Daily granularity", value=True)
    params = filters.to_params() | {"granularity": "day" if daily_view else "month"}
    try:
        data = fetch_json("/metrics/sales-trends", params)
    except Exception as exc:
        st.error(f"Unable to load sales trends: {exc}")
        return
    store_rows = data.get("store_rows", [])
    summary_rows = data.get("summary_rows", [])

    store_df = pd.DataFrame(store_rows)
    summary_df = pd.DataFrame(summary_rows)

    if store_df.empty and summary_df.empty:
        st.info("No data for selected range")
        return

    if not store_df.empty:
        store_df["period_start"] = pd.to_datetime(
            store_df["period_start"], errors="coerce"
        )
        pivot = (
            store_df.pivot_table(
                index="period_start",
                columns="store_name",
                values="total_sales",
                aggfunc="sum",
            )
            .fillna(0)
            .sort_index()
        )
    else:
        pivot = pd.DataFrame()

    if pivot.empty:
        st.info("No store sales to display")
        return

    def format_period(series: pd.Series) -> pd.Series:
        if daily_view:
            return series.dt.strftime("%d %b '%y")
        return series.dt.strftime("%b'%y")

    period_labels = format_period(pivot.index.to_series())

    store_totals = pivot.sum(axis=0).sort_values(ascending=False)
    sorted_stores = store_totals.index.tolist()
    pivot = pivot[sorted_stores]
    color_cycle = [hex_code for _, hex_code in COLOR_SEQUENCE]
    color_map = {
        store: color_cycle[idx % len(color_cycle)]
        for idx, store in enumerate(sorted_stores)
    }

    fig = go.Figure()
    for store in sorted_stores:
        values = pivot[store].values
        fig.add_bar(
            name=store,
            x=period_labels,
            y=values,
            marker_color=color_map[store],
            customdata=build_hover_data(list(values)),
            hovertemplate=f"{store}: %{{customdata[0]}}<extra></extra>",
        )

    if not summary_df.empty:
        summary_df["period_start"] = pd.to_datetime(
            summary_df["period_start"], errors="coerce"
        )
        summary_df.sort_values("period_start", inplace=True)
        summary_df["period_label"] = format_period(summary_df["period_start"])
        fig.add_scatter(
            name="Average order value",
            mode="lines+markers",
            x=summary_df["period_label"],
            y=summary_df["avg_order_value"],
            marker_color=AVG_LINE_COLOR,
            yaxis="y2",
            customdata=build_hover_data(list(summary_df["avg_order_value"])),
            hovertemplate="Average order value: %{customdata[0]}<extra></extra>",
        )

    fig.update_layout(
        title="Sales by Store and Period",
        barmode="stack",
        xaxis_title="Period",
        yaxis={"title": "Sales (₣)"},
        yaxis2={
            "title": "Average order value (₣)",
            "overlaying": "y",
            "side": "right",
        },
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02},
        template="plotly_white",
    )
    st.plotly_chart(fig, use_container_width=True)


def render_top_stores(filters: Filters):
    params = filters.to_params() | {"limit": 10}
    try:
        data = fetch_json("/metrics/top-stores", params)
    except Exception as exc:
        st.error(f"Unable to load top stores: {exc}")
        return
    rows = data.get("rows", [])
    if not rows:
        st.info("No store data for selected range")
        return
    df = pd.DataFrame(rows)
    df["gross_profit_pct"] = df.apply(
        lambda r: (r["total_gp"] / r["total_sales"]) * 100
        if r["total_sales"]
        else None,
        axis=1,
    )
    df_display = pd.DataFrame(
        {
            "Store": df["store_name"].fillna("Unknown"),
            "Average order value (₣)": df["avg_order_value"].apply(format_currency),
            "Total sales (₣)": df["total_sales"].apply(format_currency),
            "Gross profit (₣)": df["total_gp"].apply(format_currency),
            "Gross profit, %": df["gross_profit_pct"].apply(format_percentage),
        }
    )
    st.dataframe(df_display, use_container_width=True)


def _build_icon_link_html() -> str:
    """Embed small icons as data URIs so they always render."""
    icon_links = {
        "Github": ("images/github_icon.png", "https://github.com/AntonMiniazev"),
        "LinkedIn": (
            "images/LinkedIn_icon.png",
            "https://www.linkedin.com/in/antonminiazev/",
        ),
    }
    tags: List[str] = []
    for label, (relative_path, href) in icon_links.items():
        file_path = Path(__file__).resolve().parent.parent / relative_path
        try:
            encoded = base64.b64encode(file_path.read_bytes()).decode("utf-8")
        except OSError:
            continue
        src = f"data:image/png;base64,{encoded}"
        tags.append(
            f'<a href="{href}" target="_blank" rel="noopener noreferrer">'
            f'<img src="{src}" alt="{label}" /></a>'
        )
    return "".join(tags)


def main():
    st.markdown(
        """
    <style>
    body, div, p, span, h1, h2, h3 { font-family: 'Calibri', sans-serif; }
    h1 { margin-top: 0rem !important; margin-bottom: 0.5rem !important; }
    .block-container { padding-top: 1rem; }
    div[data-testid="stMetricValue"] {
        font-size: 1.6rem;  /* adjust higher/lower */
        font-weight: 600;
    }
    div[data-testid="stMetricLabel"] {
        font-size: 0.9rem;
        text-transform: uppercase;
    }
    .header-row { display: flex; align-items: center; justify-content: space-between; }
    .external-links { display: flex; gap: 8px; align-items: center; }
    .external-links img { width: 35px; height: 35px; cursor: pointer; }
    </style>
    """,
        unsafe_allow_html=True,
    )
    link_html = _build_icon_link_html()
    st.markdown(
        f"""
        <div class="header-row">
            <h1>Ampere Retail Performance Dashboard</h1>
            <div class="external-links">{link_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    filters = render_filters()
    st.divider()
    render_metrics(filters)
    st.divider()
    st.subheader("Sales Trend")
    render_sales_trend(filters)
    st.subheader("Top Performing Stores")
    render_top_stores(filters)


if __name__ == "__main__":
    main()
