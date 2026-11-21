
import json
import pandas as pd
from pathlib import Path

INPUT_FILE = Path("data/raw/lihkg_tencent_test.json")
OUTPUT_FILE = Path("data/processed/lihkg_tencent_posts.csv")

def process_lihkg():
    with open(INPUT_FILE, "r", encoding="utf-8") as file:
        posts = json.load(file)

    df = pd.DataFrame(posts)

    # Preserve original timestamps in UTC
    df["created_at"] = pd.to_datetime(
        df["created_at"], utc=True, errors="coerce"
    )
    df["last_reply_at"] = pd.to_datetime(
        df["last_reply_at"], utc=True, errors="coerce"
    )

    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=10)

    df["within_10_days"] = df["created_at"] >= cutoff
    df["ticker"] = "0700.HK"
    df["exchange"] = "Hong Kong"
    df["source"] = "LIHKG"

    columns = [
        "ticker", "exchange", "source",
        "thread_id", "title", "content",
        "author", "reply_count", "like_count",
        "dislike_count", "created_at",
        "last_reply_at", "thread_url",
        "within_10_days"
    ]

    for column in columns:
        if column not in df.columns:
            df[column] = None

    result = df[columns].copy()

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    print("\nLIHKG DATA PROCESSING")
    print("=" * 40)
    print("Total records:", len(result))
    print("Created within last 10 days:", int(result["within_10_days"].sum()))
    print("Older records:", int((~result["within_10_days"]).sum()))
    print("Saved to:", OUTPUT_FILE.resolve())

    print("\nSample:")
    print(result[["thread_id", "title", "created_at", "like_count",
                  "reply_count", "within_10_days"]].head().to_string(index=False))


if __name__ == "__main__":
    process_lihkg()
