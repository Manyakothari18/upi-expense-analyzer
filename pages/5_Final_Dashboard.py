
from pathlib import Path
import sys

import pandas as pd
import plotly.express as px
import streamlit as st

# ==========================================
# PROJECT PATH AND ML IMPORTS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.clustering import perform_clustering
from src.anomaly_detection import detect_anomalies
from src.expense_prediction import predict_expenses


# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="UPI Financial Dashboard",
    page_icon="💰",
    layout="wide"
)

st.title("💰 UPI Smart Financial Dashboard")
st.caption(
    "Expense Analytics | Spending Insights | "
    "Machine Learning | Financial Recommendations"
)


# ==========================================
# DATA LOADING
# ==========================================

@st.cache_data
def load_data():
    path = BASE_DIR / "data" / "cleaned_transactions.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"Transaction dataset not found: {path}"
        )

    data = pd.read_csv(path)

    data["timestamp"] = pd.to_datetime(
        data["timestamp"],
        errors="coerce"
    )

    data["amount"] = pd.to_numeric(
        data["amount"],
        errors="coerce"
    )

    data = data.dropna(
        subset=["timestamp", "amount"]
    )

    return data


try:
    df = load_data()
except (FileNotFoundError, KeyError, ValueError) as error:
    st.error(f"Unable to load transaction data: {error}")
    st.stop()

if df.empty:
    st.warning("No valid transactions found.")
    st.stop()


# ==========================================
# SIDEBAR FILTERS
# ==========================================

st.sidebar.header("Dashboard Filters")

# Only successful debit transactions are expenses.
expense_df = df[
    (df["status"] == "Success")
    & (df["transaction_type"] == "Debit")
    & (df["amount"] >= 0)
].copy()

if expense_df.empty:
    st.warning("No successful debit transactions available.")
    st.stop()

categories = sorted(
    expense_df["category"].dropna().unique().tolist()
)

selected_categories = st.sidebar.multiselect(
    "Select Categories",
    options=categories,
    default=categories
)

min_date = expense_df["timestamp"].min().date()
max_date = expense_df["timestamp"].max().date()

date_range = st.sidebar.date_input(
    "Select Date Range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

if len(date_range) != 2:
    st.info("Please select both start and end dates.")
    st.stop()

start_date, end_date = date_range

filtered_df = expense_df[
    (expense_df["category"].isin(selected_categories))
    & (expense_df["timestamp"].dt.date >= start_date)
    & (expense_df["timestamp"].dt.date <= end_date)
].copy()

if filtered_df.empty:
    st.warning(
        "No transactions found for the selected filters."
    )
    st.stop()

st.caption(
    "Financial statistics are calculated using "
    "successful debit transactions only."
)


# ==========================================
# 1. FINANCIAL OVERVIEW
# ==========================================

st.header("1. Financial Overview")

total_spending = filtered_df["amount"].sum()
total_transactions = len(filtered_df)
average_transaction = filtered_df["amount"].mean()
highest_transaction = filtered_df["amount"].max()

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Total Spending",
    f"₹{total_spending:,.2f}"
)

c2.metric(
    "Transactions",
    f"{total_transactions:,}"
)

c3.metric(
    "Average Transaction",
    f"₹{average_transaction:,.2f}"
)

c4.metric(
    "Highest Transaction",
    f"₹{highest_transaction:,.2f}"
)

st.divider()


# ==========================================
# 2. CATEGORY-WISE SPENDING
# ==========================================

st.header("2. Category-wise Spending Analysis")

category_spending = (
    filtered_df.groupby(
        "category",
        as_index=False
    )["amount"]
    .sum()
    .sort_values(
        "amount",
        ascending=False
    )
)

col1, col2 = st.columns(2)

with col1:
    fig = px.bar(
        category_spending,
        x="category",
        y="amount",
        color="category",
        title="Total Spending by Category",
        labels={
            "category": "Category",
            "amount": "Spending (₹)"
        }
    )

    st.plotly_chart(
        fig,
        width="stretch"
    )

with col2:
    fig = px.pie(
        category_spending,
        names="category",
        values="amount",
        hole=0.45,
        title="Category Spending Distribution"
    )

    st.plotly_chart(
        fig,
        width="stretch"
    )

st.divider()


# ==========================================
# 3. MONTHLY SPENDING TRENDS
# ==========================================

st.header("3. Monthly Spending Trends")

monthly = filtered_df.copy()

monthly["month"] = (
    monthly["timestamp"]
    .dt.to_period("M")
    .astype(str)
)

monthly_spending = (
    monthly.groupby(
        "month",
        as_index=False
    )["amount"]
    .sum()
    .sort_values("month")
)

fig = px.line(
    monthly_spending,
    x="month",
    y="amount",
    markers=True,
    title="Monthly UPI Spending",
    labels={
        "month": "Month",
        "amount": "Spending (₹)"
    }
)

st.plotly_chart(
    fig,
    width="stretch"
)

st.divider()


# ==========================================
# 4. SMART SPENDING INSIGHTS
# ==========================================

st.header("4. Smart Spending Insights")

top_category = category_spending.iloc[0]

top_category_name = top_category["category"]
top_category_amount = top_category["amount"]

top_percentage = (
    top_category_amount / total_spending * 100
    if total_spending > 0
    else 0
)

st.info(
    f"Your highest spending category is "
    f"**{top_category_name}**, accounting for "
    f"**{top_percentage:.1f}%** of total spending."
)

if top_percentage > 30:
    st.warning(
        f"Spending is concentrated in "
        f"{top_category_name}. Consider reviewing "
        "expenses in this category."
    )
else:
    st.success(
        "No single category accounts for more "
        "than 30% of selected spending."
    )

st.subheader("Suggested Category Budgets")

budget = category_spending.copy()

budget["suggested_budget"] = (
    budget["amount"] * 0.90
)

budget["potential_savings"] = (
    budget["amount"] * 0.10
)

budget_display = budget.rename(
    columns={
        "category": "Category",
        "amount": "Current Spending (₹)",
        "suggested_budget": "Suggested Budget (₹)",
        "potential_savings": "Potential Savings (₹)"
    }
)

# Format budget table for display
formatted_budget = budget_display.copy()

for column in formatted_budget.columns:
    if pd.api.types.is_numeric_dtype(formatted_budget[column]):
        formatted_budget[column] = formatted_budget[column].map(
            lambda value: f"₹{value:,.2f}"
        )

st.dataframe(
    formatted_budget,
    width="stretch",
    hide_index=True
)

st.caption(
    "Budgets are illustrative 10% reduction targets "
    "based on historical spending, not personalized "
    "financial advice."
)

st.divider()


# ==========================================
# 5. TRANSACTION ACTIVITY
# ==========================================

st.header("5. Transaction Activity")

merchant_spending = (
    filtered_df.groupby(
        "merchant",
        as_index=False
    )["amount"]
    .sum()
    .sort_values(
        "amount",
        ascending=False
    )
    .head(10)
)

fig = px.bar(
    merchant_spending,
    x="amount",
    y="merchant",
    orientation="h",
    title="Top 10 Merchants by Spending",
    labels={
        "amount": "Spending (₹)",
        "merchant": "Merchant"
    }
)

fig.update_layout(
    yaxis={"categoryorder": "total ascending"}
)

st.plotly_chart(
    fig,
    width="stretch"
)

st.divider()


# ==========================================
# 6. MACHINE LEARNING OVERVIEW
# ==========================================

st.header("6. Machine Learning Overview")

st.caption(
    "Interactive results using K-Means Clustering, "
    "Isolation Forest and Expense Prediction."
)

tab1, tab2, tab3 = st.tabs([
    "🎯 K-Means Clustering",
    "🚨 Anomaly Detection",
    "📈 Expense Prediction"
])


# ------------------------------------------
# TAB 1: K-MEANS CLUSTERING
# ------------------------------------------

with tab1:
    st.subheader("Transaction Clustering")

    st.write(
        "K-Means groups transactions based on "
        "transaction amount, hour and day of the week."
    )

    n_clusters = st.slider(
        "Number of Clusters",
        min_value=2,
        max_value=6,
        value=3,
        key="dashboard_clusters"
    )

    try:
        clustered, summary, score, _ = (
            perform_clustering(
                filtered_df,
                n_clusters=n_clusters
            )
        )

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Transactions Clustered",
            f"{len(clustered):,}"
        )

        c2.metric(
            "Clusters Formed",
            f"{clustered['cluster'].nunique()}"
        )

        c3.metric(
            "Silhouette Score",
            f"{score:.3f}"
        )

        fig = px.scatter(
            clustered,
            x="PCA1",
            y="PCA2",
            color="cluster_label",
            hover_data=[
                "amount",
                "category",
                "merchant"
            ],
            title="Transaction Clusters (PCA Projection)",
            labels={
                "PCA1": "Principal Component 1",
                "PCA2": "Principal Component 2",
                "cluster_label": "Spending Group"
            },
            opacity=0.65
        )

        st.plotly_chart(
            fig,
            width="stretch"
        )

        st.info(
            "PCA is used for 2D visualization. "
            "K-Means is trained on standardized "
            "transaction features."
        )

        st.subheader("Cluster Summary")

        st.dataframe(
            summary,
            width="stretch",
            hide_index=True
        )

        fig = px.bar(
            summary,
            x="cluster_label",
            y="transactions",
            color="cluster_label",
            title="Transactions by Cluster"
        )

        st.plotly_chart(
            fig,
            width="stretch"
        )

    except (ValueError, KeyError) as error:
        st.warning(
            f"Clustering unavailable: {error}"
        )


# ------------------------------------------
# TAB 2: ANOMALY DETECTION
# ------------------------------------------

with tab2:
    st.subheader("Unusual Transaction Detection")

    st.write(
        "Isolation Forest identifies transactions "
        "with unusual spending characteristics."
    )

    contamination = st.slider(
        "Expected Anomaly Percentage",
        min_value=1,
        max_value=15,
        value=5,
        step=1,
        key="dashboard_contamination"
    ) / 100

    try:
        if len(filtered_df) < 10:
            st.warning(
                "At least 10 transactions are required "
                "for anomaly detection."
            )
        else:
            anomalies, count, percentage, _ = (
                detect_anomalies(
                    filtered_df,
                    contamination=contamination
                )
            )

            c1, c2, c3 = st.columns(3)

            c1.metric(
                "Transactions Analyzed",
                f"{len(anomalies):,}"
            )

            c2.metric(
                "Unusual Transactions",
                f"{count:,}"
            )

            c3.metric(
                "Anomaly Percentage",
                f"{percentage:.2f}%"
            )

            fig = px.scatter(
                anomalies,
                x="timestamp",
                y="amount",
                color="anomaly",
                hover_data=[
                    "category",
                    "merchant"
                ],
                title="Normal vs Unusual Transactions",
                labels={
                    "timestamp": "Transaction Date",
                    "amount": "Amount (₹)",
                    "anomaly": "Classification"
                }
            )

            st.plotly_chart(
                fig,
                width="stretch"
            )

            st.subheader("Top Unusual Transactions")

            unusual = (
                anomalies[
                    anomalies["is_anomaly"]
                ]
                .sort_values("anomaly_score")
                .head(10)
            )

            display_columns = [
                "timestamp",
                "amount",
                "category",
                "merchant",
                "anomaly_score"
            ]

            st.dataframe(
                unusual[display_columns],
                width="stretch",
                hide_index=True
            )

            st.caption(
                "Anomalies are statistical outliers, "
                "not confirmed fraudulent transactions."
            )

    except (ValueError, KeyError, ZeroDivisionError) as error:
        st.warning(
            f"Anomaly detection unavailable: {error}"
        )


# ------------------------------------------
# TAB 3: EXPENSE PREDICTION
# ------------------------------------------

with tab3:
    st.subheader("Next Month Expense Forecast")

    st.info(
        "Forecasting uses the complete historical "
        "dataset rather than the selected dashboard "
        "date and category filters."
    )

    try:
        (
            monthly_history,
            test_predictions,
            metrics,
            best_model,
            forecast_month,
            forecast_amount
        ) = predict_expenses(expense_df)

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Forecast Month",
            forecast_month
        )

        c2.metric(
            "Predicted Expenses",
            f"₹{forecast_amount:,.2f}"
        )

        c3.metric(
            "Selected Model",
            best_model
        )

        st.subheader("Model Evaluation")

        st.dataframe(
            metrics,
            width="stretch",
            hide_index=True
        )

        if (metrics["R2"] < 0).all():
            st.warning(
                "Both prediction models have negative "
                "R² scores. Forecast accuracy is limited."
            )

        st.subheader("Actual vs Predicted Expenses")

        comparison = test_predictions.copy()

        comparison["year_month"] = (
            comparison["year_month"].astype(str)
        )

        comparison = comparison.rename(
            columns={
                "amount": "Actual Expenses"
            }
        )

        plot_df = comparison.melt(
            id_vars="year_month",
            value_vars=[
                "Actual Expenses",
                "Linear Regression",
                "Random Forest"
            ],
            var_name="Model",
            value_name="Expenses"
        )

        fig = px.line(
            plot_df,
            x="year_month",
            y="Expenses",
            color="Model",
            markers=True,
            title="Actual vs Predicted Monthly Expenses",
            labels={
                "year_month": "Month",
                "Expenses": "Expenses (₹)"
            }
        )

        st.plotly_chart(
            fig,
            width="stretch"
        )

        st.caption(
            "Forecasts are experimental and based on "
            "synthetic transaction data. Predictions "
            "should not be used for real financial decisions."
        )

    except (ValueError, KeyError) as error:
        st.warning(
            f"Expense prediction unavailable: {error}"
        )

st.divider()


# ==========================================
# 7. DOWNLOAD FINANCIAL REPORT
# ==========================================

st.header("7. Download Financial Report")

report = filtered_df.sort_values(
    "timestamp",
    ascending=False
)

st.download_button(
    "📥 Download Filtered Transactions CSV",
    data=report.to_csv(
        index=False
    ).encode("utf-8"),
    file_name="upi_financial_report.csv",
    mime="text/csv"
)

st.success(
    "Financial dashboard generated successfully!"
)
