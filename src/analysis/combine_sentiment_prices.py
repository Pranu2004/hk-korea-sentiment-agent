
from pathlib import Path
import pandas as pd

PRICE_FILE = "data/processed/stock_features.csv"
SENTIMENT_FILE = "data/processed/lihkg_sentiment.csv"
OUTPUT_FILE = "data/processed/lihkg_price_features.csv"
ALIGNED_FILE = "data/processed/lihkg_aligned_posts.csv"


def get_market_close_utc(ticker, trading_date):
    # Hong Kong closes at 16:00 HKT = 08:00 UTC.
    if ticker.endswith(".HK"):
        return pd.Timestamp(
            f"{trading_date} 08:00:00", tz="UTC"
        )

    # South Korea closes at 15:30 KST = 06:30 UTC.
    if ticker.endswith(".KS"):
        return pd.Timestamp(
            f"{trading_date} 06:30:00", tz="UTC"
        )

    return None


def main():
    prices = pd.read_csv(PRICE_FILE)
    sentiment = pd.read_csv(SENTIMENT_FILE)

    prices = prices.rename(columns={"Ticker": "ticker"})

    prices["Date"] = pd.to_datetime(
        prices["Date"], errors="coerce"
    ).dt.date

    sentiment["created_at"] = pd.to_datetime(
        sentiment["created_at"], utc=True, errors="coerce"
    )

    sentiment = sentiment.dropna(
        subset=["ticker", "created_at", "sentiment_score"]
    ).copy()

    # Find the first available trading-session close after each post.
    trading_dates = {
        ticker: sorted(group["Date"].dropna().unique())
        for ticker, group in prices.groupby("ticker")
    }

    aligned_dates = []

    for _, post in sentiment.iterrows():
        ticker = post["ticker"]
        published_at = post["created_at"]
        assigned_date = None

        for trading_date in trading_dates.get(ticker, []):
            close_time = get_market_close_utc(
                ticker, trading_date
            )

            if close_time is not None and published_at < close_time:
                assigned_date = trading_date
                break

        aligned_dates.append(assigned_date)

    sentiment["Date"] = aligned_dates

    aligned = sentiment.dropna(subset=["Date"]).copy()

    daily = (
        aligned.groupby(["ticker", "Date"])
        .agg(
            social_sentiment=("sentiment_score", "mean"),
            social_post_count=("sentiment_score", "count"),
            social_positive_count=(
                "sentiment_label",
                lambda x: (x == "positive").sum()
            ),
            social_negative_count=(
                "sentiment_label",
                lambda x: (x == "negative").sum()
            ),
            social_likes=("like_count", "sum"),
            social_replies=("reply_count", "sum"),
        )
        .reset_index()
    )

    combined = prices.merge(
        daily,
        on=["ticker", "Date"],
        how="left"
    )

    Path("data/processed").mkdir(parents=True, exist_ok=True)

    aligned.to_csv(
        ALIGNED_FILE, index=False, encoding="utf-8-sig"
    )
    combined.to_csv(
        OUTPUT_FILE, index=False, encoding="utf-8-sig"
    )

    print("\nLIHKG MARKET-CLOSE ALIGNMENT")
    print("=" * 45)
    print("Original posts:", len(sentiment))
    print("Posts assigned to a trading session:", len(aligned))
    print("Posts outside available price history:", len(sentiment) - len(aligned))
    print("Daily sentiment records:", len(daily))
    print("Price feature rows:", len(prices))
    print("Rows with matched sentiment:", combined["social_sentiment"].notna().sum())

    print("\nMatched posts by ticker:")
    print(aligned.groupby("ticker").size())

    print("\nSaved:", ALIGNED_FILE)
    print("Saved:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
