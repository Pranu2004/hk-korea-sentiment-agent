import pandas as pd
import numpy as np

from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error


FEATURES = [
    "Ticker",
    "daily_return",
    "ma_5_gap",
    "ma_10_gap",
    "volatility_5",
    "volatility_10",
    "support_gap",
    "resistance_gap"
]

NUMERIC_FEATURES = FEATURES[1:]


def train_and_evaluate(data):

    data = data.copy()
    data["Date"] = pd.to_datetime(data["Date"])
    data = data.sort_values("Date").reset_index(drop=True)

    # Predict next-day percentage return, not raw closing price.
    data["target_return"] = (
        data["target_next_close"] / data["Close"]
    ) - 1

    # Convert price indicators into relative features.
    data["ma_5_gap"] = data["ma_5"] / data["Close"] - 1
    data["ma_10_gap"] = data["ma_10"] / data["Close"] - 1
    data["support_gap"] = data["support_20"] / data["Close"] - 1
    data["resistance_gap"] = data["resistance_20"] / data["Close"] - 1

    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.dropna(subset=["target_return", "Close"])

    # Chronological split: older dates train, newer dates test.
    unique_dates = sorted(data["Date"].unique())
    split_index = int(len(unique_dates) * 0.80)
    split_date = unique_dates[split_index]

    train = data[data["Date"] < split_date].copy()
    test = data[data["Date"] >= split_date].copy()

    X_train = train[FEATURES]
    y_train = train["target_return"]

    X_test = test[FEATURES]
    y_test = test["target_return"]

    preprocessing = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]),
                NUMERIC_FEATURES
            ),
            (
                "ticker",
                OneHotEncoder(handle_unknown="ignore"),
                ["Ticker"]
            )
        ]
    )

    model = Pipeline([
        ("preprocessing", preprocessing),
        ("regressor", Ridge(alpha=1.0))
    ])

    model.fit(X_train, y_train)

    predicted_returns = model.predict(X_test)

    # Convert predicted returns into predicted closing prices.
    current_prices = X_test["Ticker"].index.map(
        lambda i: test.loc[i, "Close"]
    ).to_numpy()

    predicted_prices = current_prices * (1 + predicted_returns)
    actual_prices = test["target_next_close"].to_numpy()

    # Previous-close baseline: assume zero return.
    baseline_prices = current_prices.copy()

    # Overall return metrics.
    return_mae = mean_absolute_error(y_test, predicted_returns)
    return_rmse = np.sqrt(
        mean_squared_error(y_test, predicted_returns)
    )

    # Overall price metrics.
    price_mae = mean_absolute_error(actual_prices, predicted_prices)
    price_rmse = np.sqrt(
        mean_squared_error(actual_prices, predicted_prices)
    )

    baseline_mae = mean_absolute_error(actual_prices, baseline_prices)
    baseline_rmse = np.sqrt(
        mean_squared_error(actual_prices, baseline_prices)
    )

    actual_direction = np.sign(y_test.to_numpy())
    predicted_direction = np.sign(predicted_returns)

    directional_accuracy = np.mean(
        actual_direction == predicted_direction
    ) * 100

    metrics = {
        "training_rows": len(train),
        "testing_rows": len(test),
        "split_date": str(pd.Timestamp(split_date).date()),
        "return_mae_pct": return_mae * 100,
        "return_rmse_pct": return_rmse * 100,
        "model_price_mae": price_mae,
        "model_price_rmse": price_rmse,
        "baseline_price_mae": baseline_mae,
        "baseline_price_rmse": baseline_rmse,
        "model_directional_accuracy_pct": directional_accuracy,
        "baseline_directional_accuracy": "N/A (zero-movement baseline)"
    }

    results = test[
        ["Date", "Ticker", "Close", "target_next_close"]
    ].copy()

    results["actual_return_pct"] = y_test.to_numpy() * 100
    results["predicted_return_pct"] = predicted_returns * 100
    results["predicted_next_close"] = predicted_prices
    results["baseline_prediction"] = baseline_prices

    return model, metrics, results


if __name__ == "__main__":

    input_file = Path("data/processed/stock_features.csv")
    data = pd.read_csv(input_file)

    model, metrics, results = train_and_evaluate(data)

    output_dir = Path("reports")
    output_dir.mkdir(exist_ok=True)

    results.to_csv(
        output_dir / "model_predictions.csv",
        index=False
    )

    pd.DataFrame([metrics]).to_csv(
        output_dir / "model_metrics.csv",
        index=False
    )

    # Per-stock performance.
    per_stock = []

    for ticker, group in results.groupby("Ticker"):

        actual = group["target_next_close"].to_numpy()
        predicted = group["predicted_next_close"].to_numpy()
        baseline = group["baseline_prediction"].to_numpy()

        actual_direction = np.sign(
            actual / group["Close"].to_numpy() - 1
        )
        predicted_direction = np.sign(
            group["predicted_return_pct"].to_numpy()
        )

        per_stock.append({
            "Ticker": ticker,
            "Test_Rows": len(group),
            "Model_MAE": mean_absolute_error(actual, predicted),
            "Model_RMSE": np.sqrt(mean_squared_error(actual, predicted)),
            "Baseline_MAE": mean_absolute_error(actual, baseline),
            "Baseline_RMSE": np.sqrt(mean_squared_error(actual, baseline)),
            "Directional_Accuracy_Pct": np.mean(
                actual_direction == predicted_direction
            ) * 100
        })

    per_stock_df = pd.DataFrame(per_stock)
    per_stock_df.to_csv(
        output_dir / "per_stock_metrics.csv",
        index=False
    )

    print("\nRETURN-BASED MODEL EVALUATION")
    print("=" * 55)

    for name, value in metrics.items():
        print(f"{name}: {value}")

    print("\nPER-STOCK PERFORMANCE")
    print(per_stock_df.round(4).to_string(index=False))

    print("\nSaved:")
    print("reports/model_predictions.csv")
    print("reports/model_metrics.csv")
    print("reports/per_stock_metrics.csv")
