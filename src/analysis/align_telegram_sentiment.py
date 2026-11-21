import pandas as pd
from pathlib import Path

POSTS = Path("data/processed/telegram_sentiment.csv")
PRICES = Path("data/raw/all_stock_prices.csv")
OUTPUT = Path("data/processed/telegram_aligned_posts.csv")
DAILY = Path("data/processed/telegram_daily_sentiment.csv")

def main():
    posts = pd.read_csv(POSTS)
    prices = pd.read_csv(PRICES)

    posts["date_utc"] = pd.to_datetime(posts["date_utc"], utc=True)
    prices["Date"] = pd.to_datetime(prices["Date"]).dt.date
    prices["Ticker"] = prices["Ticker"].astype(str)

    # Only use posts with a verified ticker match.
    matched_posts = posts[posts["ticker"].notna()].copy()

    trading_dates = {
        ticker: sorted(group["Date"].unique())
        for ticker, group in prices[
            prices["Ticker"].str.endswith(".KS")
        ].groupby("Ticker")
    }

    aligned_dates = []

    for _, post in matched_posts.iterrows():
        timestamp = post["date_utc"]
        eligible_date = timestamp.date()

        # Korean stock market closing time: 06:30 UTC.
        if timestamp.time() > pd.Timestamp("06:30:00").time():
            eligible_date = (
                timestamp + pd.Timedelta(days=1)
            ).date()

        available = [
            date for date in trading_dates.get(post["ticker"], [])
            if date >= eligible_date
        ]

        aligned_dates.append(available[0] if available else None)

    matched_posts["trading_date"] = aligned_dates
    matched_posts["alignment_status"] = matched_posts[
        "trading_date"
    ].apply(
        lambda x: "aligned" if pd.notna(x) else "pending_price_bar"
    )

    matched_posts.to_csv(
        OUTPUT, index=False, encoding="utf-8-sig"
    )

    aligned = matched_posts[
        matched_posts["alignment_status"] == "aligned"
    ].copy()

    if not aligned.empty:
        daily = aligned.groupby(
            ["ticker", "trading_date"], as_index=False
        ).agg(
            post_count=("sentiment", "count"),
            average_sentiment=("sentiment_score", "mean"),
            positive_posts=("sentiment", lambda x: (x == "positive").sum()),
            negative_posts=("sentiment", lambda x: (x == "negative").sum()),
            neutral_posts=("sentiment", lambda x: (x == "neutral").sum()),
            total_views=("views", "sum"),
            total_reactions=("reactions", "sum")
        )
    else:
        daily = pd.DataFrame(columns=[
            "ticker", "trading_date", "post_count",
            "average_sentiment", "positive_posts",
            "negative_posts", "neutral_posts",
            "total_views", "total_reactions"
        ])

    daily.to_csv(DAILY, index=False, encoding="utf-8-sig")

    print("\nTelegram alignment completed.")
    print("Verified stock posts:", len(matched_posts))
    print("\nAlignment status:")
    print(
        matched_posts["alignment_status"].value_counts().to_string()
        if not matched_posts.empty else "No matched posts"
    )
    print("\nDaily sentiment:")
    print(daily.to_string(index=False))
    print("\nSaved:", OUTPUT)
    print("Saved:", DAILY)

if __name__ == "__main__":
    main()
