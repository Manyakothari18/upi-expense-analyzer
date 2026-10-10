
from pathlib import Path
import sys

import pandas as pd
import plotly.express as px
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.clustering import perform_clustering

st.set_page_config(
    page_title="ML Clustering",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Machine Learning — K-Means Clustering")
st.caption("Discover transaction patterns using unsupervised learning")

@st.cache_data
def load_data():
    path = BASE_DIR / "data" / "cleaned_transactions.csv"
    return pd.read_csv(path)

df = load_data()

n_clusters = st.slider(
    "Select Number of Clusters",
    min_value=2,
    max_value=6,
    value=3
)

with st.spinner("Training K-Means model..."):
    clustered, summary, score, model = perform_clustering(
        df, n_clusters
    )

col1, col2, col3 = st.columns(3)

col1.metric("Transactions Analyzed", len(clustered))
col2.metric("Clusters", n_clusters)
col3.metric("Silhouette Score", f"{score:.3f}")

st.divider()

st.subheader("Cluster Visualization")

fig = px.scatter(
    clustered,
    x="PCA1",
    y="PCA2",
    color="cluster_label",
    hover_data=["amount", "category", "merchant"],
    title="Transaction Clusters (PCA Projection)",
    labels={
        "PCA1": "Principal Component 1",
        "PCA2": "Principal Component 2",
        "cluster_label": "Cluster"
    },
    opacity=0.65
)

st.plotly_chart(fig, width="stretch")

st.info(
    "PCA reduces the feature space to two dimensions "
    "for visualization. K-Means clustering is performed "
    "on the standardized original features."
)

st.divider()

st.subheader("Cluster Summary")

st.dataframe(
    summary,
    width="stretch",
    hide_index=True
)

st.divider()

st.subheader("Transactions by Cluster")

fig = px.bar(
    summary,
    x="cluster_label",
    y="transactions",
    color="cluster_label",
    title="Cluster Size Distribution"
)

st.plotly_chart(fig, width="stretch")

st.subheader("Average Transaction Amount by Cluster")

fig = px.bar(
    summary,
    x="cluster_label",
    y="average_amount",
    color="cluster_label",
    title="Average Spending by Cluster",
    labels={
        "average_amount": "Average Amount (₹)",
        "cluster_label": "Cluster"
    }
)

st.plotly_chart(fig, width="stretch")

st.success("K-Means clustering completed successfully!")
