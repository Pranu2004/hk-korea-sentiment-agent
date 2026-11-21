
from pathlib import Path

import pandas as pd
from snownlp import SnowNLP


INPUT_FILE = "data/processed/lihkg_recent_10d.csv"
OUTPUT_FILE = "data/processed/lihkg_sentiment.csv"


def analyze_sentiment(text):
    if pd.isna(text) or not str(text).strip():
        return pd.Series({
            "sentiment_score": None,
            "sentiment_label": "unavailable"
        })

    try:
        score = SnowNLP(str(text)).sentiments

        if score >= 0.60:
            label = "positive"
        elif score <= 0.40:
            label = "negative"
        else:
            label = "neutral"

        return pd.Series({
            "sentiment_score": score,
            "sentiment_label": label
        })

    except Exception:
        return pd.Series({
            "sentiment_score": None,
            "sentiment_label": "unavailable"
        })


def main():
    df = pd.read_csv(INPUT_FILE)

    # Use thread title and available text.
    text_columns = ["title", "content", "own_text"]
    for column in text_columns:
        if column not in df.columns:
            df[column] = ""

    df["analysis_text"] = ""

    for column in text_columns:
        df[column] = df[column].astype("string").fillna("")
        df["analysis_text"] = df["analysis_text"] + " " + df[column].astype(str)

    df["analysis_text"] = df["analysis_text"].str.strip()

    results = df["analysis_text"].apply(analyze_sentiment)
    df = pd.concat([df, results], axis=1)

    # Engagement is recorded separately; it does not determine sentiment.
    for column in ["like_count", "reply_count"]:
        if column not in df.columns:
            df[column] = 0
        df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0)

    Path("data/processed").mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    print("\nLIHKG SENTIMENT ANALYSIS")
    print("=" * 40)
    print("Total recent posts:", len(df))
    print("Analyzed posts:", df["sentiment_score"].notna().sum())
    print("\nSentiment distribution:")
    print(df["sentiment_label"].value_counts(dropna=False))

    print("\nAverage sentiment by company:")
    print(
        df.groupby("stock_name")["sentiment_score"]
        .mean()
        .round(3)
    )

    print("\nSaved:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
