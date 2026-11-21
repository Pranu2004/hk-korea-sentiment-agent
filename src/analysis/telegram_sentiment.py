import json
import re
import pandas as pd
from pathlib import Path
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline

INPUT = Path("data/raw/telegram_market_posts.json")
OUTPUT = Path("data/processed/telegram_sentiment.csv")

STOCKS = {
    "0700.HK": [r"\b0700\.hk\b", r"\b00700\b", "tencent", "腾讯"],
    "9988.HK": [r"\b9988\.hk\b", r"\b09988\b", "alibaba", "阿里巴巴"],
    "3690.HK": [r"\b3690\.hk\b", r"\b03690\b", "meituan", "美团"],
    "1810.HK": [r"\b1810\.hk\b", r"\b01810\b", "xiaomi", "小米"],
    "0941.HK": [r"\b0941\.hk\b", r"\b00941\b", "china mobile", "中国移动"],
    "005930.KS": [r"\b005930\b", "samsung electronics", "삼성전자"],
    "000660.KS": [r"\b000660\b", "sk hynix", "sk하이닉스", "하이닉스"],
    "005380.KS": [r"\b005380\b", "hyundai motor", "현대자동차", "현대차"],
    "373220.KS": [r"\b373220\b", "lg energy solution", "lg에너지솔루션"],
    "035420.KS": [r"\b035420\b", "naver corporation", "네이버 주가", "네이버 실적"],
}

def find_tickers(text):
    text = re.sub(r"https?://\S+", "", text).lower()
    matches = []

    for ticker, keywords in STOCKS.items():
        for keyword in keywords:
            if keyword.startswith(r"\b"):
                matched = bool(re.search(keyword, text))
            else:
                matched = keyword.lower() in text

            if matched:
                matches.append(ticker)
                break

    return matches

def main():
    with INPUT.open("r", encoding="utf-8") as file:
        posts = json.load(file)

    print("Loading Korean financial sentiment model...")
    korean_model = pipeline(
        "text-classification",
        model="snunlp/KR-FinBert-SC",
        device=-1
    )
    english_model = SentimentIntensityAnalyzer()

    rows = []

    for post in posts:
        text = str(post.get("text") or "").strip()

        if not text:
            continue

        has_korean = bool(re.search(r"[가-힣]", text))

        if has_korean:
            result = korean_model(text[:1500])[0]
            label = result["label"].lower()
            confidence = float(result["score"])

            if "positive" in label:
                sentiment, score = "positive", 1.0
            elif "negative" in label:
                sentiment, score = "negative", -1.0
            else:
                sentiment, score = "neutral", 0.0

            method = "KR-FinBERT-SC"
        else:
            result = english_model.polarity_scores(text)
            score = float(result["compound"])
            sentiment = (
                "positive" if score >= 0.05 else
                "negative" if score <= -0.05 else
                "neutral"
            )
            confidence = None
            method = "VADER"

        tickers = find_tickers(text) or [None]

        for ticker in tickers:
            rows.append({
                "channel": post.get("channel"),
                "channel_title": post.get("channelTitle"),
                "post_id": post.get("postId"),
                "post_url": post.get("postUrl"),
                "date_utc": post.get("date"),
                "text": text,
                "ticker": ticker,
                "match_status": "matched" if ticker else "unmatched",
                "sentiment": sentiment,
                "sentiment_score": score,
                "model_confidence": confidence,
                "sentiment_method": method,
                "views": post.get("views", 0),
                "reactions": post.get("reactionsTotal", 0)
            })

    df = pd.DataFrame(rows)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT, index=False, encoding="utf-8-sig")

    print("\nTelegram processing completed.")
    print("Processed rows:", len(df))
    print("Matched rows:", (df["match_status"] == "matched").sum())
    print("Unmatched rows:", (df["match_status"] == "unmatched").sum())
    print("\nMatched tickers:")
    print(df[df["ticker"].notna()]["ticker"].value_counts().to_string())
    print("\nSentiment distribution:")
    print(df["sentiment"].value_counts().to_string())
    print("\nSaved:", OUTPUT)

if __name__ == "__main__":
    main()
