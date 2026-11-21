import pandas as pd
from pathlib import Path
from transformers import pipeline

INPUT = Path("data/processed/naver_korea_recent_10d.csv")
OUTPUT = Path("data/processed/naver_korea_sentiment.csv")
MODEL_NAME = "snunlp/KR-FinBert-SC"

def main():
    print("Loading Naver posts...")
    df = pd.read_csv(INPUT)

    df["post_text"] = df["post_text"].fillna("").astype(str)
    df = df[df["post_text"].str.strip() != ""].copy()

    print("Posts to analyze:", len(df))
    print("Loading Korean financial sentiment model...")

    classifier = pipeline(
        "text-classification",
        model=MODEL_NAME,
        device=-1
    )

    texts = df["post_text"].tolist()

    print("Analyzing Korean posts...")
    results = classifier(
        texts,
        batch_size=8,
        truncation=True,
        max_length=256
    )

    df["sentiment"] = [result["label"].lower() for result in results]
    df["sentiment_confidence"] = [result["score"] for result in results]

    df.to_csv(OUTPUT, index=False, encoding="utf-8-sig")

    print("\nSentiment analysis completed!")
    print("\nSentiment distribution:")
    print(df["sentiment"].value_counts().to_string())

    print("\nAverage sentiment confidence:")
    print(round(df["sentiment_confidence"].mean(), 4))

    print("\nSentiment by company:")
    print(
        pd.crosstab(df["stockName"], df["sentiment"]).to_string()
    )

    print("\nSaved:", OUTPUT)

if __name__ == "__main__":
    main()
