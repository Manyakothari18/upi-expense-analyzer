
from pathlib import Path
import sys

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.expense_prediction import predict_expenses

st.set_page_config(
    page_title="UPI Expense Prediction",
    page_icon="📈",
    layout="wide"
)

st.title("📈 UPI Expense Prediction")
st.caption(
    "Machine Learning Forecasting using "
    "Linear Regression and Random Forest"
)

st.info(
    "Predict future UPI spending based on historical "
    "monthly transaction patterns. "
    "Predictions are estimates based on synthetic data."
)


@st.cache_data
def load_data():
    file_path = BASE_DIR / "data" / "cleaned_transactions.csv"
    return pd.read_csv(file_path)


@st.cache_data(show_spinner=False)
def run_prediction(df):
    return predict_expenses(df)


df = load_data()

with st.spinner("Training prediction models..."):
    (
        monthly,
        predictions,
        metrics,
        best_model,
        next_month,
        forecast
    ) = run_prediction(df)


# -----------------------------
# Financial Prediction Overview
# -----------------------------

st.subheader("1. Expense Forecast Overview")

latest_expense = float(monthly["amount"].iloc[-1])
average_expense = float(monthly["amount"].mean())

change_percentage = (
    ((forecast - latest_expense) / latest_expense) * 100
    if latest_expense > 0 else 0
)

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Next Month Forecast",
    f"₹{forecast:,.2f}"
)

col2.metric(
    "Last Month Expenses",
    f"₹{latest_expense:,.2f}"
)

col3.metric(
    "Average Monthly Expenses",
    f"₹{average_expense:,.2f}"
)

col4.metric(
    "Expected Change",
    f"{change_percentage:+.2f}%"
)

st.success(
    f"Best Model: {best_model} | "
    f"Forecast Month: {next_month}"
)

st.divider()


# -----------------------------
# Historical Spending
# -----------------------------

st.subheader("2. Historical Monthly Spending")

history = monthly.copy()
history["Month"] = history["year_month"].astype(str)

fig = px.line(
    history,
    x="Month",
    y="amount",
    markers=True,
    title="Historical Monthly UPI Expenses",
    labels={
        "amount": "Total Expenses (₹)",
        "Month": "Month"
    }
)

fig.update_layout(
    xaxis_tickangle=-45,
    hovermode="x unified"
)

st.plotly_chart(fig, width="stretch")

st.divider()


# -----------------------------
# Model Evaluation
# -----------------------------

st.subheader("3. Machine Learning Model Comparison")

st.write(
    "The models are evaluated using the last four "
    "months as unseen test data."
)

display_metrics = metrics.copy()

for column in ["MAE", "RMSE", "R2"]:
    display_metrics[column] = display_metrics[column].round(3)

st.dataframe(
    display_metrics,
    width="stretch",
    hide_index=True
)

col1, col2 = st.columns(2)

with col1:
    fig = px.bar(
        metrics,
        x="Model",
        y="MAE",
        color="Model",
        title="Mean Absolute Error Comparison",
        labels={"MAE": "MAE (₹)"}
    )
    st.plotly_chart(fig, width="stretch")

with col2:
    fig = px.bar(
        metrics,
        x="Model",
        y="RMSE",
        color="Model",
        title="Root Mean Squared Error Comparison",
        labels={"RMSE": "RMSE (₹)"}
    )
    st.plotly_chart(fig, width="stretch")

st.caption(
    "Lower MAE and RMSE indicate smaller prediction errors. "
    "R² closer to 1 indicates a better fit; negative R² "
    "means the model performs worse than predicting "
    "the test-set mean."
)

st.divider()


# -----------------------------
# Actual vs Predicted
# -----------------------------

st.subheader("4. Actual vs Predicted Expenses")

comparison = predictions.copy()
comparison["Month"] = comparison["year_month"].astype(str)

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=comparison["Month"],
        y=comparison["amount"],
        mode="lines+markers",
        name="Actual Expenses",
        line=dict(color="#38bdf8", width=3)
    )
)

fig.add_trace(
    go.Scatter(
        x=comparison["Month"],
        y=comparison["Linear Regression"],
        mode="lines+markers",
        name="Linear Regression",
        line=dict(color="#f59e0b", dash="dash")
    )
)

fig.add_trace(
    go.Scatter(
        x=comparison["Month"],
        y=comparison["Random Forest"],
        mode="lines+markers",
        name="Random Forest",
        line=dict(color="#22c55e", dash="dot")
    )
)

fig.update_layout(
    title="Model Predictions on Test Data",
    xaxis_title="Month",
    yaxis_title="Expenses (₹)",
    hovermode="x unified"
)

st.plotly_chart(fig, width="stretch")

st.divider()


# -----------------------------
# Future Forecast
# -----------------------------

st.subheader("5. Next Month Expense Forecast")

forecast_data = pd.DataFrame({
    "Month": [
        str(monthly["year_month"].iloc[-1]),
        next_month
    ],
    "Expenses": [
        latest_expense,
        forecast
    ],
    "Type": [
        "Actual",
        "Forecast"
    ]
})

fig = px.bar(
    forecast_data,
    x="Month",
    y="Expenses",
    color="Type",
    text_auto=".2f",
    title="Last Month vs Forecast",
    labels={"Expenses": "Expenses (₹)"}
)

st.plotly_chart(fig, width="stretch")

st.warning(
    "Forecasts are experimental because the dataset "
    "contains a limited number of synthetic monthly "
    "observations. They should not be used for "
    "real financial decisions."
)

st.divider()


# -----------------------------
# Download Results
# -----------------------------

st.subheader("6. Download Prediction Results")

download_data = predictions.copy()
download_data["year_month"] = (
    download_data["year_month"].astype(str)
)

st.download_button(
    label="📥 Download Model Predictions CSV",
    data=download_data.to_csv(index=False).encode("utf-8"),
    file_name="upi_expense_predictions.csv",
    mime="text/csv"
)

st.success("Expense Prediction Analysis Completed!")
