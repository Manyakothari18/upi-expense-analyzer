
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


def perform_clustering(df, n_clusters=3):
    # Select successful debit transactions
    data = df[
        (df["status"] == "Success")
        & (df["transaction_type"] == "Debit")
    ].copy()

    # Clean and validate data
    data["amount"] = pd.to_numeric(
        data["amount"], errors="coerce"
    )
    data["timestamp"] = pd.to_datetime(
        data["timestamp"], errors="coerce"
    )

    data = data.dropna(subset=["amount", "timestamp"])
    data = data[data["amount"] >= 0].copy()

    if len(data) < n_clusters + 1:
        raise ValueError(
            "Not enough valid transactions for clustering."
        )

    # Extract transaction features
    data["hour"] = data["timestamp"].dt.hour
    data["day_number"] = data["timestamp"].dt.dayofweek

    # Reduce influence of very large transactions
    data["log_amount"] = np.log1p(data["amount"])

    features = [
        "log_amount",
        "hour",
        "day_number"
    ]

    X = data[features].copy()

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train K-Means model
    model = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=10
    )

    data["cluster"] = model.fit_predict(X_scaled)

    # PCA for 2D visualization
    pca = PCA(n_components=2)
    coordinates = pca.fit_transform(X_scaled)

    data["PCA1"] = coordinates[:, 0]
    data["PCA2"] = coordinates[:, 1]

    # Calculate silhouette score safely
    unique_clusters = data["cluster"].nunique()

    if 2 <= unique_clusters < len(data):
        score = silhouette_score(
            X_scaled,
            data["cluster"],
            sample_size=min(2000, len(data)),
            random_state=42
        )
    else:
        score = float("nan")

    # Generate cluster summary
    summary = (
        data.groupby("cluster")
        .agg(
            transactions=("amount", "count"),
            average_amount=("amount", "mean"),
            median_amount=("amount", "median"),
            total_spending=("amount", "sum")
        )
        .reset_index()
        .sort_values("average_amount")
    )

    # Assign meaningful group labels
    labels = [
        f"Spending Group {i + 1}"
        for i in range(len(summary))
    ]

    label_map = dict(zip(summary["cluster"], labels))

    data["cluster_label"] = data["cluster"].map(label_map)
    summary["cluster_label"] = summary["cluster"].map(label_map)

    return data, summary, score, model


if __name__ == "__main__":
    DATA_PATH = (
        Path(__file__).resolve().parent.parent
        / "data"
        / "cleaned_transactions.csv"
    )

    df = pd.read_csv(DATA_PATH)

    clustered, summary, score, model = perform_clustering(df)

    print("\n===== K-MEANS CLUSTERING RESULTS =====")
    print(summary.to_string(index=False))
    print("\nSilhouette Score:", round(score, 4))
