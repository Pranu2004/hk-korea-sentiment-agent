
import json
from pathlib import Path

import pandas as pd

def main():
    # Load the original Tencent dataset
    with open(
        "data/raw/lihkg_tencent_api.json",
        encoding="utf-8"
    ) as file:
        tencent = json.load(file)

    tencent_df = pd.DataFrame(tencent)
    tencent_df["ticker"] = "0700.HK"
    tencent_df["stock_name"] = "Tencent"
    tencent_df["exchange"] = "Hong Kong"
    tencent_df["source"] = "LIHKG"

    # Load the other four companies
    other_df = pd.read_csv(
        "data/raw/lihkg_hk_stocks.csv"
    )

    # Combine
    df = pd.concat(
        [tencent_df, other_df],
        ignore_index=True
    )

    # Parse timestamps as UTC
    df["created_at"] = pd.to_datetime(
        df["created_at"], utc=True, errors="coerce"
    )
    df["last_reply_at"] = pd.to_datetime(
        df["last_reply_at"], utc=True, errors="coerce"
    )

    # Remove duplicate threads for the same stock
    df = df.drop_duplicates(
        subset=["ticker", "thread_id"]
    )

    # Calculate recent-post and recent-activity flags
    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=10)

    df["created_within_10_days"] = df["created_at"] >= cutoff
    df["active_within_10_days"] = df["last_reply_at"] >= cutoff

    # Save complete dataset
    Path("data/processed").mkdir(
        parents=True, exist_ok=True
    )

    all_file = "data/processed/lihkg_all_stocks.csv"
    recent_file = "data/processed/lihkg_recent_10d.csv"

    df.to_csv(
        all_file, index=False, encoding="utf-8-sig"
    )

    recent = df[df["created_within_10_days"]].copy()

    recent.to_csv(
        recent_file, index=False, encoding="utf-8-sig"
    )

    # Print summary
    print("\nLIHKG COMBINED DATA SUMMARY")
    print("=" * 45)
    print("Total unique stock-thread records:", len(df))
    print("Posts created in last 10 days:", len(recent))
    print(
        "Threads active in last 10 days:",
        int(df["active_within_10_days"].sum())
    )

    print("\nRecords by company:")
    print(df.groupby("stock_name").size())

    print("\nRecent posts by company:")
    print(
        recent.groupby("stock_name").size()
        .reindex(df["stock_name"].dropna().unique(), fill_value=0)
    )

    print("\nSaved:", all_file)
    print("Saved:", recent_file)


if __name__ == "__main__":
    main()
