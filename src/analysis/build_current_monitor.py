import pandas as pd
from pathlib import Path

PRICES = Path("data/raw/all_stock_prices.csv")
SENTIMENT = Path("data/processed/combined_daily_sentiment.csv")
PENDING = Path("data/processed/pending_social_posts.csv")

OUTPUT = Path("data/processed/current_sentiment_monitor.csv")

def main():
    prices = pd.read_csv(PRICES)
    sentiment = pd.read_csv(SENTIMENT)
    pending = pd.read_csv(PENDING)

    prices["Date"] = pd.to_datetime(prices["Date"]).dt.date
    prices["Ticker"] = prices["Ticker"].astype(str)

    sentiment["Date"] = pd.to_datetime(
        sentiment["Date"]
    ).dt.date
    sentiment["ticker"] = sentiment["ticker"].astype(str)

    # Merge historical, price-aligned sentiment with market prices.
    monitor = prices.merge(
        sentiment,
        left_on=["Ticker", "Date"],
        right_on=["ticker", "Date"],
        how="left"
    )

    monitor = monitor.drop(
        columns=["ticker"], errors="ignore"
    )

    monitor.to_csv(
        OUTPUT, index=False, encoding="utf-8-sig"
    )

    print("\nLatest available market date:")
    print(
        monitor.groupby("Ticker")["Date"].max().to_string()
    )

    print("\nHistorical sentiment source coverage:")
    for column in [
        "news_count", "lihkg_count",
        "naver_count", "telegram_count"
    ]:
        if column in monitor.columns:
            print(
                column, ":",
                int(monitor[column].fillna(0).sum())
            )

    print("\nLatest available Naver-aligned sentiment:")
    if "naver_count" in monitor.columns:
        naver_rows = monitor[
            monitor["naver_count"].fillna(0) > 0
        ]
        if not naver_rows.empty:
            print(naver_rows[
                ["Ticker", "Date", "Close", "naver_count",
                 "naver_sentiment", "naver_avg_confidence"]
            ].to_string(index=False))
        else:
            print("No Naver sentiment in current price monitor.")

    print("\nPending social posts:")
    if not pending.empty:
        columns = [
            col for col in [
                "channel", "ticker", "sentiment",
                "sentiment_score", "model_confidence",
                "views", "reactions", "date_utc",
                "alignment_status"
            ] if col in pending.columns
        ]
        print(pending[columns].to_string(index=False))
    else:
        print("No pending social posts.")

    print("\nTotal market rows:", len(monitor))
    print("Saved:", OUTPUT)


if __name__ == "__main__":
    main()
