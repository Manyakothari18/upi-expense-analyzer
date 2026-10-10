
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "cleaned_transactions.csv"


def analyze_transactions(df):
    # Only successful transactions count towards financial totals
    successful = df[df["status"] == "Success"]

    expenses = successful[
        successful["transaction_type"] == "Debit"
    ].copy()

    income = successful[
        successful["transaction_type"] == "Credit"
    ].copy()

    total_expenses = expenses["amount"].sum()
    total_income = income["amount"].sum()

    print("\n========== UPI EXPENSE ANALYSIS ==========")

    print("Total Transactions:", len(df))
    print("Successful Transactions:", len(successful))
    print("Total Expenses: Rs.", round(total_expenses, 2))
    print("Total Income: Rs.", round(total_income, 2))
    print("Net Cash Flow: Rs.",
          round(total_income - total_expenses, 2))

    print("Average Expense: Rs.",
          round(expenses["amount"].mean(), 2))

    print("Maximum Expense: Rs.",
          round(expenses["amount"].max(), 2))

    print("\n========== CATEGORY-WISE SPENDING ==========")

    category_spending = (
        expenses.groupby("category")["amount"]
        .sum()
        .sort_values(ascending=False)
    )
    print(category_spending.round(2))

    print("\n========== MONTHLY SPENDING ==========")

    monthly_spending = (
        expenses.groupby("year_month")["amount"]
        .sum()
        .sort_index()
    )
    print(monthly_spending.round(2))

    print("\n========== DESCRIPTIVE STATISTICS ==========")

    print(expenses["amount"].describe().round(2))

    print("\n========== TOP 5 MERCHANTS ==========")

    top_merchants = (
        expenses.groupby("merchant")["amount"]
        .sum()
        .sort_values(ascending=False)
        .head(5)
    )
    print(top_merchants.round(2))


if __name__ == "__main__":
    df = pd.read_csv(DATA_PATH)
    analyze_transactions(df)
