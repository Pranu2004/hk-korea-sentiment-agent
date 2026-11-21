import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path

st.set_page_config(
    page_title="HK & Korea Market Sentiment",
    page_icon="📊",
    layout="wide"
)

DATA_FILE = Path("data/processed/current_sentiment_monitor.csv")
PENDING_FILE = Path("data/processed/pending_social_posts.csv")

st.title("📊 Hong Kong & South Korea Stock Sentiment Monitor")
st.caption("Market prices, financial news and social discussion")

if not DATA_FILE.exists():
    st.error(f"Dataset not found: {DATA_FILE}")
    st.stop()


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_FILE)
    df["Date"] = pd.to_datetime(df["Date"])

    if "exchange" in df.columns and "Exchange" in df.columns:
        df["Market"] = df["exchange"].combine_first(df["Exchange"])
    elif "exchange" in df.columns:
        df["Market"] = df["exchange"]
    else:
        df["Market"] = df["Exchange"]

    df["Market"] = df["Market"].fillna(
        df["Ticker"].astype(str).apply(
            lambda x: "Hong Kong" if x.endswith(".HK") else "South Korea"
        )
    )

    count_columns = [
        "news_count", "lihkg_count", "naver_count", "telegram_count",
        "news_positive", "news_negative", "news_neutral",
        "lihkg_positive", "lihkg_negative", "lihkg_neutral",
        "naver_positive", "naver_negative", "naver_neutral",
        "telegram_positive", "telegram_negative", "telegram_neutral"
    ]

    for col in count_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col], errors="coerce"
            ).fillna(0)

    # Add missing source columns safely.
    for col in [
        "news_count", "lihkg_count", "naver_count", "telegram_count"
    ]:
        if col not in df.columns:
            df[col] = 0

    df = df.sort_values(["Ticker", "Date"]).copy()

    # Calculate support and resistance using past prices only.
    df["Support_20"] = (
        df.groupby("Ticker")["Low"]
        .transform(
            lambda s: s.shift(1).rolling(20, min_periods=5).min()
        )
    )

    df["Resistance_20"] = (
        df.groupby("Ticker")["High"]
        .transform(
            lambda s: s.shift(1).rolling(20, min_periods=5).max()
        )
    )

    # Combine only sentiment sources with actual aligned observations.
    source_info = [
        ("news_sentiment", "news_count"),
        ("lihkg_sentiment", "lihkg_count"),
        ("naver_sentiment", "naver_count"),
        ("telegram_sentiment", "telegram_count")
    ]

    weighted_total = pd.Series(0.0, index=df.index)
    total_items = pd.Series(0.0, index=df.index)

    for score_col, count_col in source_info:
        if score_col not in df.columns:
            continue

        scores = pd.to_numeric(df[score_col], errors="coerce")
        counts = df[count_col].fillna(0)
        valid = scores.notna() & (counts > 0)

        weighted_total += (
            scores.fillna(0) * counts.where(valid, 0)
        )
        total_items += counts.where(valid, 0)

    df["Combined_Sentiment"] = (
        weighted_total / total_items.replace(0, np.nan)
    )
    df["Total_Items"] = total_items

    def sentiment_label(row):
        if row["Total_Items"] == 0:
            return "No recent data"
        if row["Combined_Sentiment"] >= 0.10:
            return "Positive"
        if row["Combined_Sentiment"] <= -0.10:
            return "Negative"
        return "Neutral"

    df["Sentiment_Direction"] = df.apply(
        sentiment_label, axis=1
    )

    return df


@st.cache_data
def load_pending():
    if not PENDING_FILE.exists():
        return pd.DataFrame()

    pending = pd.read_csv(PENDING_FILE)

    if "date_utc" in pending.columns:
        pending["date_utc"] = pd.to_datetime(
            pending["date_utc"], utc=True, errors="coerce"
        )

    return pending


df = load_data()
pending = load_pending()

# Latest available market record for every stock.
latest = (
    df.sort_values("Date")
    .groupby("Ticker", as_index=False)
    .tail(1)
    .copy()
)

latest["Mention_Count"] = (
    latest["news_count"] +
    latest["lihkg_count"] +
    latest["naver_count"] +
    latest["telegram_count"]
)

latest = latest.sort_values(
    "Mention_Count", ascending=False
)

st.sidebar.header("Dashboard filters")

markets = ["All Markets"] + sorted(
    latest["Market"].dropna().unique().tolist()
)
selected_market = st.sidebar.selectbox(
    "Select exchange", markets
)

if selected_market != "All Markets":
    latest = latest[
        latest["Market"] == selected_market
    ]
    available_tickers = latest["Ticker"].tolist()
    history = df[df["Market"] == selected_market]
else:
    available_tickers = latest["Ticker"].tolist()
    history = df.copy()

if not available_tickers:
    st.warning("No stocks available for this filter.")
    st.stop()

selected_ticker = st.sidebar.selectbox(
    "Select stock", available_tickers
)

selected_row = latest[
    latest["Ticker"] == selected_ticker
].iloc[0]

stock_history = history[
    history["Ticker"] == selected_ticker
].sort_values("Date")

currency = (
    "HKD" if selected_ticker.endswith(".HK") else "KRW"
)

st.subheader(
    f"{selected_row['Company']} ({selected_ticker})"
)

st.caption(
    f"Exchange: {selected_row['Market']} | "
    f"Latest price date: {selected_row['Date'].date()}"
)

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Latest closing price",
    f"{selected_row['Close']:,.2f} {currency}"
)

c2.metric(
    "Aligned news and social items",
    f"{int(selected_row['Mention_Count'])}"
)

c3.metric(
    "Sentiment direction",
    selected_row["Sentiment_Direction"]
)

score = selected_row["Combined_Sentiment"]
c4.metric(
    "Combined sentiment score",
    "N/A" if pd.isna(score) else f"{score:.3f}"
)

st.divider()

# ---------------- PRICE CHART ----------------
st.subheader("Historical stock price")

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=stock_history["Date"],
    y=stock_history["Close"],
    mode="lines",
    name="Closing price"
))

fig.add_trace(go.Scatter(
    x=stock_history["Date"],
    y=stock_history["Support_20"],
    mode="lines",
    name="20-session support",
    line=dict(dash="dash")
))

fig.add_trace(go.Scatter(
    x=stock_history["Date"],
    y=stock_history["Resistance_20"],
    mode="lines",
    name="20-session resistance",
    line=dict(dash="dash")
))

fig.update_layout(
    xaxis_title="Date",
    yaxis_title=f"Price ({currency})",
    hovermode="x unified",
    height=450
)

st.plotly_chart(fig, use_container_width=True)

# ---------------- SOURCE COVERAGE ----------------
st.subheader("Sentiment source coverage")

source_values = {
    "Financial news (Exa)": int(selected_row["news_count"]),
    "LIHKG discussions": int(selected_row["lihkg_count"]),
    "Naver discussions": int(selected_row["naver_count"]),
    "Telegram posts": int(selected_row["telegram_count"])
}

st.bar_chart(
    pd.DataFrame({
        "Source": list(source_values.keys()),
        "Items": list(source_values.values())
    }).set_index("Source")
)

st.caption(
    "Counts represent items aligned to the selected stock's latest "
    "market date. Zero means no aligned observations, not neutral sentiment."
)

# ---------------- PENDING POSTS ----------------
st.subheader("Pending social posts")

if not pending.empty:
    selected_pending = pending[
        pending["ticker"].astype(str) == selected_ticker
    ].copy()

    if not selected_pending.empty:
        st.info(
            f"{len(selected_pending)} post(s) mention this stock but "
            "are waiting for a matching future trading-price bar. "
            "They are excluded from historical sentiment calculations."
        )

        display_columns = [
            col for col in [
                "channel", "ticker", "date_utc", "sentiment",
                "sentiment_score", "model_confidence",
                "views", "reactions", "alignment_status", "post_url"
            ]
            if col in selected_pending.columns
        ]

        st.dataframe(
            selected_pending[display_columns],
            use_container_width=True,
            hide_index=True
        )
    else:
        st.caption(
            "No pending social posts for the selected stock."
        )
else:
    st.caption("No pending social posts are currently recorded.")

# ---------------- STOCK COMPARISON ----------------
st.subheader("Stock comparison")

comparison = latest[[
    "Company", "Ticker", "Market", "Date", "Close",
    "Mention_Count", "Sentiment_Direction",
    "Combined_Sentiment", "Support_20", "Resistance_20"
]].copy()

comparison = comparison.rename(columns={
    "Date": "Latest Date",
    "Close": (
        f"Close ({currency})"
        if selected_market != "All Markets"
        else "Close (local currency)"
    ),
    "Mention_Count": "Aligned Items",
    "Combined_Sentiment": "Sentiment Score",
    "Support_20": "20-session Support",
    "Resistance_20": "20-session Resistance"
})

st.dataframe(
    comparison.sort_values(
        "Aligned Items", ascending=False
    ),
    use_container_width=True,
    hide_index=True
)

csv_data = comparison.to_csv(
    index=False
).encode("utf-8")

st.download_button(
    "Download stock comparison CSV",
    data=csv_data,
    file_name="stock_sentiment_comparison.csv",
    mime="text/csv"
)

# ---------------- METHODOLOGY ----------------
st.divider()
st.subheader("Data and methodology notes")

st.markdown("""
- Exa financial news sentiment uses VADER, a general-language baseline.
- LIHKG sentiment uses SnowNLP.
- Naver discussion sentiment uses KR-FinBERT-SC.
- Telegram Korean sentiment uses KR-FinBERT-SC; English uses VADER.
- Combined sentiment is weighted by aligned item counts.
- Pending posts are displayed separately and are not used as historical signals.
- Missing source coverage is not treated as neutral sentiment.
- Support and resistance use the previous 20 trading sessions.
- Sentiment direction is descriptive, not a guaranteed price prediction.
- The price-only Ridge model did not consistently outperform the
  previous-close baseline in out-of-sample testing.
""")

st.warning(
    "Research and demonstration dashboard only. "
    "Sentiment and technical levels are not investment advice."
)

st.caption(
    "Data source: collected project datasets | "
    "Latest available market data: November 20, 2026"
)
