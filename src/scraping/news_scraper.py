
import os
import pandas as pd

from datetime import datetime, timedelta, timezone
from pathlib import Path
from dotenv import load_dotenv
from exa_py import Exa


load_dotenv()

STOCKS = {
    "0700.HK": ("Tencent", "Hong Kong"),
    "9988.HK": ("Alibaba", "Hong Kong"),
    "3690.HK": ("Meituan", "Hong Kong"),
    "1810.HK": ("Xiaomi", "Hong Kong"),
    "0941.HK": ("China Mobile", "Hong Kong"),
    "005930.KS": ("Samsung Electronics", "South Korea"),
    "000660.KS": ("SK Hynix", "South Korea"),
    "005380.KS": ("Hyundai Motor", "South Korea"),
    "373220.KS": ("LG Energy Solution", "South Korea"),
    "035420.KS": ("NAVER", "South Korea")
}


def fetch_financial_news():
    api_key = os.getenv("EXA_API_KEY")

    if not api_key:
        raise ValueError("EXA_API_KEY is missing from .env")

    exa = Exa(api_key=api_key)

    # Rolling 10-day UTC publication window
    start_date = (
        datetime.now(timezone.utc) - timedelta(days=10)
    ).isoformat(timespec="seconds").replace("+00:00", "Z")

    all_articles = []

    for ticker, (company, exchange) in STOCKS.items():

        query = (
            f"{company} {ticker} stock market financial news"
        )

        print(f"\nSearching news for {company} ({ticker})...")

        try:
            response = exa.search(
                query,
                category="news",
                start_published_date=start_date,
                num_results=10,
                contents={"highlights": True}
            )

            for article in response.results:

                highlights = article.highlights or []

                article_text = " ".join(highlights)

                all_articles.append({
                    "ticker": ticker,
                    "company": company,
                    "exchange": exchange,
                    "title": article.title or "",
                    "text": article_text,
                    "published_date": article.published_date or "",
                    "author": article.author or "",
                    "url": article.url,
                    "source": "Exa"
                })

            print(f"Articles found: {len(response.results)}")

        except Exception as error:
            print(f"News search failed for {ticker}: {error}")

    return all_articles


if __name__ == "__main__":

    articles = fetch_financial_news()

    output_dir = Path("data/raw")
    output_dir.mkdir(parents=True, exist_ok=True)

    if articles:
        news_df = pd.DataFrame(articles)

        news_df = news_df.drop_duplicates(subset=["url"])

        output_file = output_dir / "financial_news.csv"
        news_df.to_csv(output_file, index=False)

        print("\nNEWS DOWNLOAD SUMMARY")
        print("=" * 45)
        print("Total unique articles:", len(news_df))
        print("Hong Kong articles:",
              len(news_df[news_df["exchange"] == "Hong Kong"]))
        print("South Korea articles:",
              len(news_df[news_df["exchange"] == "South Korea"]))
        print("Saved to:", output_file.resolve())

        print("\nSample articles:")
        print(
            news_df[["ticker", "title", "published_date"]]
            .head(10)
            .to_string(index=False)
        )

    else:
        print("\nNo articles retrieved. Check your Exa API access.")
