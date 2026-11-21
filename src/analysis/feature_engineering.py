
import pandas as pd
from pathlib import Path


def create_features(data):
    """Create historical price features for each stock."""

    data = data.copy()

    data["Date"] = pd.to_datetime(data["Date"])
    data = data.sort_values(["Ticker", "Date"])

    grouped = data.groupby("Ticker", group_keys=False)

    # Daily percentage return
    data["daily_return"] = grouped["Close"].pct_change()

    # Moving averages
    data["ma_5"] = grouped["Close"].transform(
        lambda x: x.rolling(5).mean()
    )

    data["ma_10"] = grouped["Close"].transform(
        lambda x: x.rolling(10).mean()
    )

    # Historical volatility
    data["volatility_5"] = grouped["daily_return"].transform(
        lambda x: x.rolling(5).std()
    )

    data["volatility_10"] = grouped["daily_return"].transform(
        lambda x: x.rolling(10).std()
    )

    # Previous 20 trading days' support and resistance
    data["support_20"] = grouped["Low"].transform(
        lambda x: x.shift(1).rolling(20).min()
    )

    data["resistance_20"] = grouped["High"].transform(
        lambda x: x.shift(1).rolling(20).max()
    )

    # Predict the next trading day's closing price
    data["target_next_close"] = grouped["Close"].shift(-1)

    # Remove rows without sufficient historical features or target
    data = data.dropna().reset_index(drop=True)

    return data


if __name__ == "__main__":

    input_file = Path("data/raw/all_stock_prices.csv")
    output_file = Path("data/processed/stock_features.csv")

    stock_data = pd.read_csv(input_file)

    feature_data = create_features(stock_data)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    feature_data.to_csv(output_file, index=False)

    print("\nFEATURE ENGINEERING COMPLETED")
    print("=" * 45)
    print("Companies:", feature_data["Ticker"].nunique())
    print("Feature records:", len(feature_data))
    print("Saved to:", output_file)

    print("\nSample records:")
    print(
        feature_data[
            [
                "Ticker",
                "Date",
                "Close",
                "daily_return",
                "ma_5",
                "ma_10",
                "support_20",
                "resistance_20",
                "target_next_close"
            ]
        ].head().to_string(index=False)
    )
