
import pandas as pd
from pathlib import Path


TIMEZONES = {
    "Hong Kong": "Asia/Hong_Kong",
    "South Korea": "Asia/Seoul"
}

MARKET_CLOSE = {
    "Hong Kong": "16:00",
    "South Korea": "15:30"
}


def align_news_with_prices():

    news_file = Path("data/processed/news_sentiment.csv")
    price_file = Path("data/raw/all_stock_prices.csv")

    output_file = Path(
        "data/processed/aligned_news_sentiment.csv"
    )

    news = pd.read_csv(news_file)
    prices = pd.read_csv(price_file)

    prices["Date"] = pd.to_datetime(
        prices["Date"]
    ).dt.normalize()

    news["published_date"] = pd.to_datetime(
        news["published_date"],
        utc=True,
        errors="coerce"
    )

    news = news.dropna(subset=["published_date"])

    aligned_groups = []

    for ticker, ticker_news in news.groupby("ticker"):

        exchange = ticker_news["exchange"].iloc[0]

        timezone = TIMEZONES[exchange]
        close_time = MARKET_CLOSE[exchange]

        # Convert UTC publication timestamps to exchange time
        local_times = ticker_news[
            "published_date"
        ].dt.tz_convert(timezone)

        local_dates = local_times.dt.strftime("%Y-%m-%d")

        # Determine whether each article was published
        # before or after that day's market close
        close_minutes = (
            int(close_time.split(":")[0]) * 60
            + int(close_time.split(":")[1])
        )

        published_minutes = (
            local_times.dt.hour * 60
            + local_times.dt.minute
        )

        effective_dates = pd.to_datetime(local_dates)

        after_close = published_minutes >= close_minutes

        effective_dates.loc[after_close] += pd.Timedelta(days=1)

        ticker_news = ticker_news.copy()
        ticker_news["effective_date"] = effective_dates.values

        # Get actual available trading dates for this stock
        ticker_prices = prices[
            prices["Ticker"] == ticker
        ][["Date"]].drop_duplicates().sort_values("Date")

        # Map each article to the next available trading session
        mapped = pd.merge_asof(
            ticker_news.sort_values("effective_date"),
            ticker_prices,
            left_on="effective_date",
            right_on="Date",
            direction="forward"
        )

        mapped = mapped.dropna(subset=["Date"])

        aligned_groups.append(mapped)

    if not aligned_groups:
        print("No news could be aligned with stock prices.")
        return

    aligned = pd.concat(aligned_groups, ignore_index=True)

    daily = (
        aligned.groupby(["ticker", "exchange", "Date"])
        .agg(
            average_sentiment=("sentiment_score", "mean"),
            article_count=("sentiment_score", "count"),
            positive_count=(
                "sentiment",
                lambda x: (x == "positive").sum()
            ),
            negative_count=(
                "sentiment",
                lambda x: (x == "negative").sum()
            ),
            neutral_count=(
                "sentiment",
                lambda x: (x == "neutral").sum()
            )
        )
        .reset_index()
    )

    daily.to_csv(output_file, index=False)

    print("\nNEWS-PRICE ALIGNMENT SUMMARY")
    print("=" * 45)
    print("Aligned articles:", len(aligned))
    print("Unique stocks:", daily["ticker"].nunique())
    print("Trading-date sentiment records:", len(daily))
    print("Saved to:", output_file.resolve())

    print("\nSample aligned records:")
    print(daily.head(10).to_string(index=False))


if __name__ == "__main__":
    align_news_with_prices()
