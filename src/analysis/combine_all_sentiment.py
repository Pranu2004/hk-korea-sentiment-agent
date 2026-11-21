import pandas as pd
from pathlib import Path

NEWS = Path("data/processed/aligned_news_sentiment.csv")
LIHKG = Path("data/processed/lihkg_aligned_posts.csv")
NAVER = Path("data/processed/naver_aligned_posts.csv")
TELEGRAM = Path("data/processed/telegram_aligned_posts.csv")
PRICES = Path("data/processed/stock_features.csv")

OUTPUT = Path("data/processed/combined_daily_sentiment.csv")
PRICE_OUTPUT = Path("data/processed/sentiment_price_features.csv")
PENDING_OUTPUT = Path("data/processed/pending_social_posts.csv")


def main():
    # ---------------- NEWS ----------------
    news = pd.read_csv(NEWS)
    news["Date"] = pd.to_datetime(news["Date"]).dt.date

    news_daily = news.rename(columns={
        "average_sentiment": "news_sentiment",
        "article_count": "news_count",
        "positive_count": "news_positive",
        "negative_count": "news_negative",
        "neutral_count": "news_neutral"
    })

    news_daily = news_daily[
        ["ticker", "Date", "news_sentiment", "news_count",
         "news_positive", "news_negative", "news_neutral"]
    ]

    # ---------------- LIHKG ----------------
    lihkg = pd.read_csv(LIHKG)
    lihkg["Date"] = pd.to_datetime(lihkg["Date"]).dt.date
    lihkg["sentiment_score"] = pd.to_numeric(
        lihkg["sentiment_score"], errors="coerce"
    )

    # Convert SnowNLP score from 0..1 to signed -1..1.
    lihkg["lihkg_signal"] = 2 * lihkg["sentiment_score"] - 1

    lihkg_daily = lihkg.groupby(
        ["ticker", "Date"], as_index=False
    ).agg(
        lihkg_sentiment=("lihkg_signal", "mean"),
        lihkg_count=("sentiment_label", "count"),
        lihkg_positive=("sentiment_label", lambda x: (x == "positive").sum()),
        lihkg_negative=("sentiment_label", lambda x: (x == "negative").sum()),
        lihkg_neutral=("sentiment_label", lambda x: (x == "neutral").sum()),
        lihkg_likes=("like_count", "sum"),
        lihkg_replies=("reply_count", "sum")
    )

    # ---------------- NAVER ----------------
    naver = pd.read_csv(NAVER)
    naver = naver[naver["alignment_status"] == "aligned"].copy()
    naver["trading_date"] = pd.to_datetime(
        naver["trading_date"]
    ).dt.date

    label_signal = {
        "positive": 1,
        "negative": -1,
        "neutral": 0
    }

    naver["naver_signal"] = (
        naver["sentiment"].str.lower().map(label_signal)
    )

    naver_daily = naver.groupby(
        ["ticker", "trading_date"], as_index=False
    ).agg(
        naver_sentiment=("naver_signal", "mean"),
        naver_count=("sentiment", "count"),
        naver_positive=("sentiment", lambda x: (x == "positive").sum()),
        naver_negative=("sentiment", lambda x: (x == "negative").sum()),
        naver_neutral=("sentiment", lambda x: (x == "neutral").sum()),
        naver_avg_confidence=("sentiment_confidence", "mean"),
        naver_likes=("likes", "sum"),
        naver_views=("views", "sum")
    ).rename(columns={"trading_date": "Date"})

    # ---------------- TELEGRAM ----------------
    telegram = pd.read_csv(TELEGRAM)

    # Keep pending posts separately; they must not enter historical features.
    pending = telegram[
        telegram["alignment_status"] != "aligned"
    ].copy()

    pending.to_csv(
        PENDING_OUTPUT, index=False, encoding="utf-8-sig"
    )

    telegram = telegram[
        telegram["alignment_status"] == "aligned"
    ].copy()

    if not telegram.empty:
        telegram["trading_date"] = pd.to_datetime(
            telegram["trading_date"]
        ).dt.date

        telegram["telegram_signal"] = (
            telegram["sentiment"].str.lower().map(label_signal)
        )

        telegram_daily = telegram.groupby(
            ["ticker", "trading_date"], as_index=False
        ).agg(
            telegram_sentiment=("telegram_signal", "mean"),
            telegram_count=("sentiment", "count"),
            telegram_positive=("sentiment", lambda x: (x == "positive").sum()),
            telegram_negative=("sentiment", lambda x: (x == "negative").sum()),
            telegram_neutral=("sentiment", lambda x: (x == "neutral").sum()),
            telegram_avg_confidence=("model_confidence", "mean"),
            telegram_views=("views", "sum"),
            telegram_reactions=("reactions", "sum")
        ).rename(columns={"trading_date": "Date"})
    else:
        telegram_daily = pd.DataFrame(columns=[
            "ticker", "Date", "telegram_sentiment",
            "telegram_count", "telegram_positive",
            "telegram_negative", "telegram_neutral",
            "telegram_avg_confidence", "telegram_views",
            "telegram_reactions"
        ])

    # ---------------- COMBINE SOURCES ----------------
    combined = news_daily.merge(
        lihkg_daily, on=["ticker", "Date"], how="outer"
    )

    combined = combined.merge(
        naver_daily, on=["ticker", "Date"], how="outer"
    )

    combined = combined.merge(
        telegram_daily, on=["ticker", "Date"], how="outer"
    )

    combined["exchange"] = combined["ticker"].apply(
        lambda x: "Hong Kong" if str(x).endswith(".HK")
        else "South Korea"
    )

    combined = combined.sort_values(
        ["exchange", "ticker", "Date"]
    )

    combined.to_csv(
        OUTPUT, index=False, encoding="utf-8-sig"
    )

    # ---------------- MERGE WITH PRICE FEATURES ----------------
    prices = pd.read_csv(PRICES)
    prices["Date"] = pd.to_datetime(prices["Date"]).dt.date
    prices["Ticker"] = prices["Ticker"].astype(str)

    price_panel = prices.merge(
        combined,
        left_on=["Ticker", "Date"],
        right_on=["ticker", "Date"],
        how="left"
    )

    price_panel = price_panel.drop(
        columns=["ticker"], errors="ignore"
    )

    count_columns = [
        "news_count", "news_positive", "news_negative", "news_neutral",
        "lihkg_count", "lihkg_positive", "lihkg_negative",
        "lihkg_neutral", "lihkg_likes", "lihkg_replies",
        "naver_count", "naver_positive", "naver_negative",
        "naver_neutral", "naver_likes", "naver_views",
        "telegram_count", "telegram_positive", "telegram_negative",
        "telegram_neutral", "telegram_views", "telegram_reactions"
    ]

    for column in count_columns:
        if column in price_panel.columns:
            price_panel[column] = (
                price_panel[column].fillna(0).astype(int)
            )

    price_panel.to_csv(
        PRICE_OUTPUT, index=False, encoding="utf-8-sig"
    )

    # ---------------- REPORT ----------------
    print("\nCombined sentiment records:", len(combined))
    print("\nRecords by exchange:")
    print(combined.groupby("exchange").size().to_string())

    print("\nSource coverage:")
    for column in [
        "news_count", "lihkg_count",
        "naver_count", "telegram_count"
    ]:
        print(
            column, ":",
            int(combined[column].fillna(0).sum())
        )

    print("\nPrice feature rows:", len(price_panel))

    for column, label in [
        ("news_count", "News"),
        ("lihkg_count", "LIHKG"),
        ("naver_count", "Naver"),
        ("telegram_count", "Telegram")
    ]:
        print(
            f"Rows with {label}:",
            int(price_panel[column].gt(0).sum())
        )

    print("\nPending social posts:", len(pending))
    print("Saved:", OUTPUT)
    print("Saved:", PRICE_OUTPUT)
    print("Saved:", PENDING_OUTPUT)


if __name__ == "__main__":
    main()
