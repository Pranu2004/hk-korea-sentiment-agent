import pandas as pd
import html
import re
from pathlib import Path

INPUT = Path("data/raw/naver_korea_posts.json")
OUTPUT = Path("data/processed")

STOCK_MAP = {
    "005930": "005930.KS",
    "000660": "000660.KS",
    "005380": "005380.KS",
    "373220": "373220.KS",
    "035420": "035420.KS",
}

def clean_text(value):
    if pd.isna(value):
        return ""
    value = html.unescape(str(value))
    value = re.sub(r"<[^>]+>", " ", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip()

def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)

    df = pd.read_json(INPUT)

    # Normalize stock codes and map them to Yahoo Finance tickers
    df["stockCode"] = df["stockCode"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(6)
    df["ticker"] = df["stockCode"].map(STOCK_MAP)
    df["exchange"] = "South Korea"

    # Convert Korean timestamps to UTC
    df["writtenAt"] = pd.to_datetime(df["writtenAt"], utc=True, errors="coerce")

    # Clean post text
    df["title"] = df["title"].apply(clean_text)
    df["text"] = df["text"].apply(clean_text)
    df["post_text"] = (df["title"] + " " + df["text"]).str.strip()

    # Convert engagement fields to numbers
    for column in ["likes", "dislikes", "views"]:
        df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0)

    # Remove invalid and duplicate records
    df = df.dropna(subset=["writtenAt", "ticker"])
    df = df.drop_duplicates(subset=["url"])

    # Keep posts from the last 10 days
    now = pd.Timestamp.now(tz="UTC")
    cutoff = now - pd.Timedelta(days=10)
    recent = df[df["writtenAt"] >= cutoff].copy()

    # Save cleaned datasets
    df.to_csv(OUTPUT / "naver_korea_all_posts.csv", index=False, encoding="utf-8-sig")
    recent.to_csv(OUTPUT / "naver_korea_recent_10d.csv", index=False, encoding="utf-8-sig")

    print("Naver data processing complete")
    print("Total cleaned posts:", len(df))
    print("Posts in last 10 days:", len(recent))
    print("\nPosts by company:")
    print(recent.groupby(["ticker", "stockName"]).size().to_string())
    print("\nDate range (UTC):")
    print("From:", recent["writtenAt"].min())
    print("To:", recent["writtenAt"].max())
    print("\nSaved:")
    print("data/processed/naver_korea_all_posts.csv")
    print("data/processed/naver_korea_recent_10d.csv")

if __name__ == "__main__":
    main()
