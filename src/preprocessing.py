
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "upi_transactions.csv"
CLEAN_PATH = BASE_DIR / "data" / "cleaned_transactions.csv"


def clean_data(df):
    df = df.copy()

    print("Original dataset shape:", df.shape)
    print("Missing values:\n", df.isnull().sum())
    print("Duplicate transaction IDs:",
          df["transaction_id"].duplicated().sum())

    # Remove duplicate transaction IDs
    df = df.drop_duplicates(subset=["transaction_id"])

    # Convert amount to numeric
    df["amount"] = pd.to_numeric(
        df["amount"], errors="coerce"
    )

    # Convert date and time into timestamp
    df["timestamp"] = pd.to_datetime(
        df["date"].astype(str) + " " + df["time"].astype(str),
        errors="coerce"
    )

    # Remove invalid transactions
    df = df.dropna(subset=["amount", "timestamp"])
    df = df[df["amount"] > 0]

    # Remove records with invalid transaction information
    df = df[
        df["transaction_type"].isin(["Debit", "Credit"])
        & df["status"].isin(["Success", "Failed", "Pending"])
    ]

    # Extract date and time features
    df["year"] = df["timestamp"].dt.year
    df["month_number"] = df["timestamp"].dt.month
    df["month"] = df["timestamp"].dt.strftime("%B")
    df["day_of_week"] = df["timestamp"].dt.strftime("%A")
    df["hour"] = df["timestamp"].dt.hour
    df["year_month"] = df["timestamp"].dt.strftime("%Y-%m")

    df = df.sort_values("timestamp").reset_index(drop=True)

    print("\nCleaned dataset shape:", df.shape)
    print("Missing values after cleaning:",
          df.isnull().sum().sum())

    return df


if __name__ == "__main__":
    df = pd.read_csv(DATA_PATH)

    cleaned_df = clean_data(df)
    cleaned_df.to_csv(CLEAN_PATH, index=False)

    print("\nCleaned dataset saved at:", CLEAN_PATH)
