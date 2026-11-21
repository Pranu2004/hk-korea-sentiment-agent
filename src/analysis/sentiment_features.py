
import pandas as pd
from pathlib import Path


def create_daily_news_features():

    input_file = Path("data/processed/news_sentiment.csv")
    output_file = Path("data/processed/daily_news_sentiment.csv")

    if not input_file.exists():
        raise FileNotFoundError("News sentiment file not found.")

    news = pd.read_csv(input_file)

    # Parse publication timestamps as UTC
    news["published_date"] = pd.to_datetime(
        news["published_date"],
        utc=True,
        errors="coerce"
    )

    news = news.dropna(subset=["published_date"])

    # Use UTC calendar dates for the initial aggregation
    news["Date"] = news["published_date"].dt.strftime("%Y-%m-%d")

    daily = (
        news.groupby(["ticker", "exchange", "Date"])
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

    output_file.parent.mkdir(parents=True, exist_ok=True)
    daily.to_csv(output_file, index=False)

    print("\nDAILY NEWS FEATURES")
    print("=" * 45)
    print("Unique stocks:", daily["ticker"].nunique())
    print("Daily stock-date records:", len(daily))
    print("Total articles:", daily["article_count"].sum())
    print("Saved to:", output_file.resolve())

    print("\nSample records:")
    print(daily.head(10).to_string(index=False))


if __name__ == "__main__":
    create_daily_news_features()
