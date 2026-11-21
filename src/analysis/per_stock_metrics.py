import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from pathlib import Path

INPUT = Path("reports/model_predictions.csv")
OUTPUT = Path("reports/per_stock_metrics.csv")

df = pd.read_csv(INPUT)

results = []

for ticker, group in df.groupby("Ticker"):
    actual = group["target_next_close"]
    predicted = group["predicted_next_close"]
    baseline = group["baseline_prediction"]
    current = group["Close"]

    actual_direction = np.sign(actual - current)
    model_direction = np.sign(predicted - current)
    baseline_direction = np.sign(baseline - current)

    results.append({
        "Ticker": ticker,
        "Test_Rows": len(group),
        "MAE": mean_absolute_error(actual, predicted),
        "RMSE": np.sqrt(mean_squared_error(actual, predicted)),
        "Baseline_MAE": mean_absolute_error(actual, baseline),
        "Baseline_RMSE": np.sqrt(mean_squared_error(actual, baseline)),
        "Directional_Accuracy_Pct": (
            actual_direction == model_direction
        ).mean() * 100,
        "Baseline_Directional_Accuracy_Pct": (
            actual_direction == baseline_direction
        ).mean() * 100
    })

metrics = pd.DataFrame(results)
metrics = metrics.sort_values("Ticker")

metrics.to_csv(OUTPUT, index=False)

print("\nPER-STOCK MODEL PERFORMANCE")
print(metrics.round(3).to_string(index=False))
print("\nSaved:", OUTPUT)
