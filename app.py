
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(
    page_title="UPI Expense Analyzer",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_PATH = (
    Path(__file__).resolve().parent
    / "data"
    / "cleaned_transactions.csv"
)


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


st.title("💳 UPI Expense Pattern Analyzer")
st.caption(
    "Interactive Data Science Dashboard | Synthetic UPI Data"
)

try:
    df = load_data()
except (FileNotFoundError, ValueError) as error:
    st.error(f"Unable to load dataset: {error}")
    st.stop()

# Sidebar filters
st.sidebar.title("Dashboard Filters")

categories = sorted(df["category"].dropna().unique())
selected_categories = st.sidebar.multiselect(
    "Select Categories",
    options=categories,
    default=categories,
)

transaction_types = st.sidebar.multiselect(
    "Transaction Type",
    options=["Debit", "Credit"],
    default=["Debit", "Credit"],
)

date_range = st.sidebar.date_input(
    "Date Range",
    value=(
        df["timestamp"].min().date(),
        df["timestamp"].max().date(),
    ),
    min_value=df["timestamp"].min().date(),
    max_value=df["timestamp"].max().date(),
)

filtered = df[
    df["category"].isin(selected_categories)
    & df["transaction_type"].isin(transaction_types)
].copy()

if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
    start_date, end_date = date_range
    filtered = filtered[
        filtered["timestamp"].dt.date.between(
            start_date, end_date
        )
    ]

if filtered.empty:
    st.warning("No transactions match the selected filters.")
    st.stop()

successful = filtered[filtered["status"] == "Success"]

expenses = successful[
    successful["transaction_type"] == "Debit"
]

income = successful[
    successful["transaction_type"] == "Credit"
]

total_expenses = expenses["amount"].sum()
total_income = income["amount"].sum()
net_cash_flow = total_income - total_expenses
avg_expense = expenses["amount"].mean()

# KPI cards
st.subheader("Financial Overview")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Expenses", f"₹{total_expenses:,.2f}")
col2.metric("Total Income", f"₹{total_income:,.2f}")
col3.metric("Net Cash Flow", f"₹{net_cash_flow:,.2f}")
col4.metric(
    "Average Expense",
    f"₹{avg_expense:,.2f}" if pd.notna(avg_expense) else "N/A",
)

st.divider()

# Category and monthly charts
left, right = st.columns(2)

with left:
    st.subheader("Category-wise Spending")

    category_data = (
        expenses.groupby("category", as_index=False)["amount"]
        .sum()
        .sort_values("amount", ascending=False)
    )

    if not category_data.empty:
        fig = px.pie(
            category_data,
            names="category",
            values="amount",
            hole=0.45,
            title="Expense Distribution",
        )
        st.plotly_chart(fig, width="stretch")
    else:
        st.info("No successful debit transactions available.")

with right:
    st.subheader("Monthly Spending Trend")

    monthly_data = (
        expenses.assign(
            year_month=expenses["timestamp"].dt.strftime("%Y-%m")
        )
        .groupby("year_month", as_index=False)["amount"]
        .sum()
        .sort_values("year_month")
    )

    if not monthly_data.empty:
        fig = px.line(
            monthly_data,
            x="year_month",
            y="amount",
            markers=True,
            title="Monthly Expenses",
            labels={
                "year_month": "Month",
                "amount": "Amount (₹)",
            },
        )
        st.plotly_chart(fig, width="stretch")
    else:
        st.info("No monthly expense data available.")

st.divider()

# Merchant spending
st.subheader("Top 10 Merchants")

merchant_data = (
    expenses.groupby("merchant", as_index=False)["amount"]
    .sum()
    .sort_values("amount", ascending=False)
    .head(10)
)

if not merchant_data.empty:
    fig = px.bar(
        merchant_data,
        x="merchant",
        y="amount",
        color="amount",
        title="Highest Spending Merchants",
        labels={
            "merchant": "Merchant",
            "amount": "Amount (₹)",
        },
    )
    st.plotly_chart(fig, width="stretch")
else:
    st.info("No merchant expense data available.")

st.divider()

# Transactions table
st.subheader("Transaction Explorer")

st.dataframe(
    filtered.sort_values("timestamp", ascending=False),
    width="stretch",
    hide_index=True,
)

csv_data = filtered.to_csv(index=False).encode("utf-8")

st.download_button(
    label="📥 Download Filtered Transactions",
    data=csv_data,
    file_name="filtered_upi_transactions.csv",
    mime="text/csv",
)

st.caption(
    "Academic project — All displayed transactions are synthetic."
)
