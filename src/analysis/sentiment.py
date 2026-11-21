
import pandas as pd
from pathlib import Path
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()


def analyze_text(text):
    """Calculate sentiment score and label."""

    if not isinstance(text, str) or not text.strip():
        return {
            "sentiment_score": 0.0,
            "sentiment": "neutral"
        }

    scores = analyzer.polarity_scores(text)
    compound = scores["compound"]

    if compound >= 0.05:
        label = "positive"
    elif compound <= -0.05:
        label = "negative"
    else:
        label = "neutral"

    return {
        "sentiment_score": compound,
        "sentiment": label
    }


def analyze_posts(posts):
    """Analyze a list of social-media posts."""

    df = pd.DataFrame(posts)

    if df.empty:
        return df

    if "text" not in df.columns:
        raise ValueError("Posts must contain a 'text' column.")

    results = df["text"].apply(analyze_text)

    df["sentiment_score"] = results.apply(
        lambda result: result["sentiment_score"]
    )

    df["sentiment"] = results.apply(
        lambda result: result["sentiment"]
    )

    return df


def analyze_news_file():

    input_file = Path("data/raw/financial_news.csv")
    output_file = Path("data/processed/news_sentiment.csv")

    if not input_file.exists():
        raise FileNotFoundError("Financial news CSV not found.")

    news = pd.read_csv(input_file).fillna("")

    # Combine article headline and available highlights
    news["text"] = (
        news["title"].astype(str) + ". " +
        news["text"].astype(str)
    )

    results = analyze_posts(news.to_dict("records"))

    output_file.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_file, index=False)

    print("\nNEWS SENTIMENT ANALYSIS")
    print("=" * 45)

    print("Total articles:", len(results))
    print("\nSentiment distribution:")
    print(results["sentiment"].value_counts())

    print("\nAverage sentiment by exchange:")
    print(
        results.groupby("exchange")["sentiment_score"]
        .mean()
    )

    print("\nSaved to:", output_file.resolve())


if __name__ == "__main__":
    analyze_news_file()
