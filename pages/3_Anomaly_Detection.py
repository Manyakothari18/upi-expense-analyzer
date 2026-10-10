
from pathlib import Path
import sys

import pandas as pd
import plotly.express as px
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.anomaly_detection import detect_anomalies

st.set_page_config(
    page_title="Anomaly Detection",
    page_icon="🚨",
    layout="wide"
)

st.title("🚨 UPI Transaction Anomaly Detection")
st.caption("Machine Learning using Isolation Forest")

st.info(
    "This model identifies unusual spending patterns. "
    "Flagged transactions are not necessarily fraudulent."
)


@st.cache_data
def load_data():
    path = BASE_DIR / "data" / "cleaned_transactions.csv"
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def analyze_data(df, contamination):
    results, count, percentage, _ = detect_anomalies(
        df, contamination
    )
    return results, count, percentage


df = load_data()

contamination = st.slider(
    "Expected Anomaly Percentage (%)",
    min_value=1,
    max_value=15,
    value=5,
    step=1
)

with st.spinner("Analyzing transactions..."):
    results, anomaly_count, anomaly_percentage = analyze_data(
        df, contamination / 100
    )

normal_count = len(results) - anomaly_count

col1, col2, col3, col4 = st.columns(4)

col1.metric("Transactions Analyzed", f"{len(results):,}")
col2.metric("Normal Transactions", f"{normal_count:,}")
col3.metric("Potential Anomalies", f"{anomaly_count:,}")
col4.metric("Anomaly Rate", f"{anomaly_percentage:.2f}%")

st.divider()

st.subheader("1. Normal vs Anomalous Transactions")

distribution = (
    results["anomaly"]
    .value_counts()
    .reset_index()
)
distribution.columns = ["Status", "Count"]

fig = px.pie(
    distribution,
    names="Status",
    values="Count",
    hole=0.5,
    color="Status",
    color_discrete_map={
        "Normal": "#16a34a",
        "Anomaly": "#ef4444"
    },
    title="Transaction Classification"
)

st.plotly_chart(fig, width="stretch")

st.divider()

st.subheader("2. Transaction Anomaly Visualization")

plot_data = results.copy()
plot_data["date"] = pd.to_datetime(plot_data["timestamp"])

fig = px.scatter(
    plot_data,
    x="date",
    y="amount",
    color="anomaly",
    color_discrete_map={
        "Normal": "#16a34a",
        "Anomaly": "#ef4444"
    },
    hover_data=[
        "transaction_id",
        "merchant",
        "category",
        "anomaly_score"
    ],
    title="Transaction Amount vs Date",
    labels={
        "date": "Transaction Date",
        "amount": "Amount (₹)",
        "anomaly": "Classification"
    },
    opacity=0.7
)

st.plotly_chart(fig, width="stretch")

st.divider()

st.subheader("3. Anomaly Score Distribution")

fig = px.histogram(
    results,
    x="anomaly_score",
    color="anomaly",
    nbins=60,
    color_discrete_map={
        "Normal": "#16a34a",
        "Anomaly": "#ef4444"
    },
    title="Isolation Forest Anomaly Scores"
)

fig.add_vline(
    x=0,
    line_dash="dash",
    annotation_text="Decision Boundary"
)

st.plotly_chart(fig, width="stretch")

st.divider()

st.subheader("4. Top Unusual Transactions")

anomalies = (
    results[results["is_anomaly"]]
    .sort_values("anomaly_score")
)

display_columns = [
    "transaction_id",
    "timestamp",
    "amount",
    "category",
    "merchant",
    "anomaly_score"
]

st.dataframe(
    anomalies[display_columns].head(25),
    width="stretch",
    hide_index=True
)

st.divider()

st.subheader("5. Download Anomaly Results")

st.download_button(
    label="📥 Download Anomaly Analysis CSV",
    data=results.to_csv(index=False).encode("utf-8"),
    file_name="upi_anomaly_results.csv",
    mime="text/csv"
)

st.success("Isolation Forest analysis completed successfully!")
