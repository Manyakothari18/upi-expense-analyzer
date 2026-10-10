
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "cleaned_transactions.csv"

def predict_expenses(df):
    data = df.copy()

    # Only successful debit transactions
    data = data[
        (data["status"] == "Success") &
        (data["transaction_type"] == "Debit")
    ].copy()

    data["date"] = pd.to_datetime(data["date"])

    # Aggregate monthly expenses
    data["year_month"] = data["date"].dt.to_period("M")

    monthly = (
        data.groupby("year_month")["amount"]
        .sum()
        .reset_index()
        .sort_values("year_month")
    )

    # Add missing months as zero-spending months
    all_months = pd.period_range(
        monthly["year_month"].min(),
        monthly["year_month"].max(),
        freq="M"
    )

    monthly = (
        monthly.set_index("year_month")
        .reindex(all_months, fill_value=0)
        .rename_axis("year_month")
        .reset_index()
    )

    monthly["month_index"] = np.arange(len(monthly))
    monthly["month_number"] = monthly["year_month"].dt.month

    # Calendar features
    monthly["month_sin"] = np.sin(
        2 * np.pi * monthly["month_number"] / 12
    )
    monthly["month_cos"] = np.cos(
        2 * np.pi * monthly["month_number"] / 12
    )

    features = ["month_index", "month_sin", "month_cos"]

    if len(monthly) < 8:
        raise ValueError(
            "At least 8 months of transaction history are required."
        )

    # Chronological split: last 4 months for testing
    train = monthly.iloc[:-4]
    test = monthly.iloc[-4:]

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=200,
            random_state=42,
            min_samples_leaf=2
        )
    }

    evaluation = []
    predictions = test[["year_month", "amount"]].copy()

    for name, model in models.items():
        model.fit(train[features], train["amount"])
        y_pred = model.predict(test[features])
        y_pred = np.maximum(y_pred, 0)

        predictions[name] = y_pred

        evaluation.append({
            "Model": name,
            "MAE": mean_absolute_error(
                test["amount"], y_pred
            ),
            "RMSE": np.sqrt(
                mean_squared_error(test["amount"], y_pred)
            ),
            "R2": r2_score(
                test["amount"], y_pred
            )
        })

    metrics = pd.DataFrame(evaluation).sort_values("MAE")

    best_model_name = metrics.iloc[0]["Model"]

    # Retrain best model on all available months
    best_model = models[best_model_name]
    best_model.fit(monthly[features], monthly["amount"])

    next_month = monthly["year_month"].iloc[-1] + 1
    next_index = len(monthly)
    next_month_number = next_month.month

    future_features = pd.DataFrame([{
        "month_index": next_index,
        "month_sin": np.sin(
            2 * np.pi * next_month_number / 12
        ),
        "month_cos": np.cos(
            2 * np.pi * next_month_number / 12
        )
    }])

    forecast = float(
        max(best_model.predict(future_features)[0], 0)
    )

    return (
        monthly,
        predictions,
        metrics,
        best_model_name,
        str(next_month),
        forecast
    )


if __name__ == "__main__":
    df = pd.read_csv(DATA_PATH)

    (
        monthly,
        predictions,
        metrics,
        best_model,
        next_month,
        forecast
    ) = predict_expenses(df)

    print("\n===== EXPENSE PREDICTION RESULTS =====")
    print("\nMonthly Data:")
    print(monthly[["year_month", "amount"]].to_string(index=False))

    print("\nModel Evaluation:")
    print(metrics.to_string(index=False))

    print("\nBest Model:", best_model)
    print("Forecast Month:", next_month)
    print(f"Predicted Expenses: Rs. {forecast:,.2f}")
