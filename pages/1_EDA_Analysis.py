
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(
    page_title="EDA Analysis",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Exploratory Data Analysis")
st.caption("Statistical analysis of synthetic UPI transactions")

DATA_PATH = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "cleaned_transactions.csv"
)


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


df = load_data()

# Financial analysis should exclude failed/pending transactions
expenses = df[
    (df["status"] == "Success")
    & (df["transaction_type"] == "Debit")
].copy()

if expenses.empty:
    st.warning("No successful debit transactions found.")
    st.stop()

st.subheader("1. Descriptive Statistics")

amount = expenses["amount"]

stats = pd.DataFrame({
    "Metric": [
        "Count", "Mean", "Median", "Mode",
        "Minimum", "Maximum", "Standard Deviation",
        "Variance", "Q1", "Q3", "IQR", "Skewness"
    ],
    "Value": [
        amount.count(),
        amount.mean(),
        amount.median(),
        amount.mode().iloc[0],
        amount.min(),
        amount.max(),
        amount.std(),
        amount.var(),
        amount.quantile(0.25),
        amount.quantile(0.75),
        amount.quantile(0.75) - amount.quantile(0.25),
        amount.skew()
    ]
})

st.dataframe(
    stats.style.format({"Value": "{:,.2f}"}),
    width="stretch",
    hide_index=True
)

st.divider()

st.subheader("2. Transaction Amount Distribution")

col1, col2 = st.columns(2)

with col1:
    fig = px.histogram(
        expenses,
        x="amount",
        nbins=50,
        title="Transaction Amount Histogram",
        labels={"amount": "Amount (₹)"}
    )
    st.plotly_chart(fig, width="stretch")

with col2:
    fig = px.box(
        expenses,
        y="amount",
        title="Transaction Amount Box Plot",
        labels={"amount": "Amount (₹)"}
    )
    st.plotly_chart(fig, width="stretch")

st.info(
    "Histogram shows how frequently different transaction "
    "amounts occur. Box plot helps identify unusually high "
    "or low transaction amounts."
)

st.divider()

st.subheader("3. Weekday Spending Analysis")

weekday_order = [
    "Monday", "Tuesday", "Wednesday", "Thursday",
    "Friday", "Saturday", "Sunday"
]

weekday_data = (
    expenses.groupby("day_of_week", as_index=False)["amount"]
    .sum()
)

fig = px.bar(
    weekday_data,
    x="day_of_week",
    y="amount",
    category_orders={"day_of_week": weekday_order},
    title="Total Spending by Weekday",
    labels={
        "day_of_week": "Day",
        "amount": "Total Spending (₹)"
    }
)

st.plotly_chart(fig, width="stretch")

st.divider()

st.subheader("4. Hourly Transaction Analysis")

expenses["hour"] = expenses["timestamp"].dt.hour

hourly_data = (
    expenses.groupby("hour", as_index=False)
    .size()
    .rename(columns={"size": "transactions"})
)

fig = px.line(
    hourly_data,
    x="hour",
    y="transactions",
    markers=True,
    title="Transaction Frequency by Hour",
    labels={
        "hour": "Hour of Day",
        "transactions": "Number of Transactions"
    }
)

st.plotly_chart(fig, width="stretch")

st.divider()

st.subheader("5. Spending Heatmap")

heatmap_data = expenses.pivot_table(
    index="day_of_week",
    columns="hour",
    values="amount",
    aggfunc="sum",
    fill_value=0
)

heatmap_data = heatmap_data.reindex(
    weekday_order, fill_value=0
)

fig = px.imshow(
    heatmap_data,
    labels={
        "x": "Hour",
        "y": "Weekday",
        "color": "Spending (₹)"
    },
    title="Weekday vs Hour Spending Heatmap",
    aspect="auto"
)

st.plotly_chart(fig, width="stretch")

st.divider()

st.subheader("6. Category-wise Spending")

category_data = (
    expenses.groupby("category", as_index=False)["amount"]
    .sum()
    .sort_values("amount", ascending=False)
)

fig = px.bar(
    category_data,
    x="category",
    y="amount",
    color="category",
    title="Total Expenses by Category",
    labels={
        "category": "Category",
        "amount": "Total Expenses (₹)"
    }
)

st.plotly_chart(fig, width="stretch")

st.success("Exploratory Data Analysis completed successfully!")
