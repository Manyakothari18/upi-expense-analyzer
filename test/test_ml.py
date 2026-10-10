
from pathlib import Path
import sys

import pandas as pd
import pytest

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.clustering import perform_clustering
from src.anomaly_detection import detect_anomalies
from src.expense_prediction import predict_expenses

DATA_PATH = BASE_DIR / "data" / "cleaned_transactions.csv"


@pytest.fixture(scope="module")
def transactions():
    df = pd.read_csv(DATA_PATH)
    assert not df.empty
    return df


def test_clustering(transactions):
    clustered, summary, score, model = perform_clustering(
        transactions, n_clusters=3
    )

    assert len(clustered) > 0
    assert clustered["cluster"].nunique() == 3
    assert len(summary) == 3
    assert -1 <= score <= 1


def test_anomaly_detection(transactions):
    results, count, percentage, model = detect_anomalies(
        transactions, contamination=0.05
    )

    assert len(results) > 0
    assert count == int(results["is_anomaly"].sum())
    assert 0 <= percentage <= 100
    assert set(results["anomaly"].unique()).issubset(
        {"Normal", "Anomaly"}
    )


def test_expense_prediction(transactions):
    monthly, predictions, metrics, best_model, next_month, forecast = (
        predict_expenses(transactions)
    )

    assert len(monthly) >= 8
    assert not predictions.empty
    assert not metrics.empty
    assert best_model in ["Linear Regression", "Random Forest"]
    assert forecast >= 0
    assert isinstance(next_month, str)
