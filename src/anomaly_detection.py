
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


def detect_anomalies(df, contamination=0.05):
    data = df[
        (df["status"] == "Success") &
        (df["transaction_type"] == "Debit")
    ].copy()

    data["timestamp"] = pd.to_datetime(data["timestamp"])
    data["hour"] = data["timestamp"].dt.hour
    data["day_number"] = data["timestamp"].dt.dayofweek

    data["log_amount"] = np.log1p(data["amount"])

    merchant_counts = data["merchant"].value_counts()
    category_counts = data["category"].value_counts()

    data["merchant_frequency"] = data["merchant"].map(
        merchant_counts
    )
    data["category_frequency"] = data["category"].map(
        category_counts
    )

    features = [
        "log_amount",
        "hour",
        "day_number",
        "merchant_frequency",
        "category_frequency"
    ]

    X = data[features]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=42,
        n_jobs=-1
    )

    predictions = model.fit_predict(X_scaled)

    data["anomaly"] = np.where(
        predictions == -1, "Anomaly", "Normal"
    )

    data["is_anomaly"] = predictions == -1
    data["anomaly_score"] = model.decision_function(X_scaled)

    anomaly_count = int(data["is_anomaly"].sum())
    anomaly_percentage = anomaly_count / len(data) * 100

    return data, anomaly_count, anomaly_percentage, model


if __name__ == "__main__":
    DATA_PATH = (
        Path(__file__).resolve().parent.parent
        / "data"
        / "cleaned_transactions.csv"
    )

    df = pd.read_csv(DATA_PATH)

    results, count, percentage, model = detect_anomalies(df)

    print("\n===== ISOLATION FOREST RESULTS =====")
    print("Transactions Analyzed:", len(results))
    print("Normal Transactions:", len(results) - count)
    print("Anomalies Detected:", count)
    print("Anomaly Percentage:", round(percentage, 2), "%")

    print("\nTop 10 Unusual Transactions:")

    columns = [
        "transaction_id",
        "amount",
        "category",
        "merchant",
        "anomaly_score"
    ]

    print(
        results[results["is_anomaly"]]
        .sort_values("anomaly_score")[columns]
        .head(10)
        .to_string(index=False)
    )
