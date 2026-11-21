
from src.analysis.sentiment import analyze_posts

sample_posts = [
    {
        "exchange": "Hong Kong",
        "ticker": "0700.HK",
        "text": "Tencent stock is performing very well. Strong growth expected!"
    },
    {
        "exchange": "Hong Kong",
        "ticker": "9988.HK",
        "text": "Alibaba stock is falling badly. Investors are worried."
    },
    {
        "exchange": "South Korea",
        "ticker": "005930.KS",
        "text": "Samsung Electronics is stable today."
    },
    {
        "exchange": "South Korea",
        "ticker": "000660.KS",
        "text": "SK Hynix has excellent growth potential!"
    }
]

results = analyze_posts(sample_posts)

print("\nSENTIMENT ANALYSIS RESULTS")
print("=" * 60)

print(
    results[
        ["exchange", "ticker", "text", "sentiment_score", "sentiment"]
    ].to_string(index=False)
)
