import pandas as pd
from pathlib import Path

POSTS = Path("data/processed/naver_korea_sentiment.csv")
PRICES = Path("data/raw/all_stock_prices.csv")
OUTPUT = Path("data/processed/naver_aligned_posts.csv")
DAILY = Path("data/processed/naver_daily_sentiment.csv")

def main():
    posts = pd.read_csv(POSTS)
    prices = pd.read_csv(PRICES)

    posts["writtenAt"] = pd.to_datetime(posts["writtenAt"], utc=True)
    prices["Date"] = pd.to_datetime(prices["Date"]).dt.date
    prices["Ticker"] = prices["Ticker"].astype(str)

    trading_dates = {
        ticker: sorted(group["Date"].unique())
        for ticker, group in prices[
            prices["Ticker"].str.endswith(".KS")
        ].groupby("Ticker")
    }

    aligned_dates = []

    for _, post in posts.iterrows():
        ticker = str(post["ticker"])
        timestamp = post["writtenAt"]
        eligible_date = timestamp.date()

        # Korean market closes at 06:30 UTC.
        # Posts after close wait for the next trading session.
        if timestamp.time() > pd.Timestamp("06:30:00").time():
            eligible_date = (timestamp + pd.Timedelta(days=1)).date()

        available = [
            date for date in trading_dates.get(ticker, [])
            if date >= eligible_date
        ]

        aligned_dates.append(available[0] if available else None)

    posts["trading_date"] = aligned_dates
    posts["alignment_status"] = posts["trading_date"].apply(
        lambda x: "aligned" if pd.notna(x) else "pending_price_bar"
    )

    posts.to_csv(OUTPUT, index=False, encoding="utf-8-sig")

    matched = posts[posts["alignment_status"] == "aligned"].copy()

    if not matched.empty:
        daily = matched.groupby(
            ["ticker", "trading_date"], as_index=False
        ).agg(
            post_count=("sentiment", "count"),
            average_sentiment_confidence=("sentiment_confidence", "mean"),
            positive_posts=("sentiment", lambda x: (x == "positive").sum()),
            negative_posts=("sentiment", lambda x: (x == "negative").sum()),
            neutral_posts=("sentiment", lambda x: (x == "neutral").sum()),
            total_likes=("likes", "sum"),
            total_views=("views", "sum")
        )

        daily.to_csv(DAILY, index=False, encoding="utf-8-sig")

        print("\nDaily sentiment summary:")
        print(daily.to_string(index=False))
    else:
        print("No posts can be aligned to available price bars yet.")

    print("\nAlignment status:")
    print(posts["alignment_status"].value_counts().to_string())

    print("\nPosts by assigned trading date:")
    print(posts.groupby(
        ["ticker", "trading_date"], dropna=False
    ).size().to_string())

    print("\nSaved:", OUTPUT)
    if not matched.empty:
        print("Saved:", DAILY)

if __name__ == "__main__":
    main()
