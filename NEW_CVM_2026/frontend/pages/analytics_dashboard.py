"""
frontend/pages/analytics_dashboard.py
=====================================
Page 4: Visualise dataset statistics with interactive Plotly charts.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import io
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.api_client import get_dataset_stats, download_synthetic


def render():
    st.header("Analytics Dashboard")
    st.markdown("Explore dataset statistics and visualisations for uploaded or generated datasets.")

    default_id = st.session_state.get("last_dataset_id", "")
    dataset_id = st.text_input("Dataset ID", value=default_id, placeholder="e.g. ds_ebcca060")

    data_source = st.radio(
        "Analyse",
        ["Synthetic Dataset", "Original Dataset"],
        horizontal=True,
    )

    if st.button("Load Analytics", type="primary", use_container_width=True):
        if not dataset_id.strip():
            st.error("Please enter a Dataset ID.")
            return

        with st.spinner("Loading dataset statistics..."):
            try:
                stats = get_dataset_stats(dataset_id.strip())
            except Exception as e:
                st.error(f"Failed to load stats: {e}")
                return

            # Try to load actual data for charting
            df = None
            try:
                if data_source == "Synthetic Dataset":
                    csv_bytes = download_synthetic(dataset_id.strip())
                    df = pd.read_csv(io.BytesIO(csv_bytes))
                else:
                    # Try to read original from datasets/original/
                    path = os.path.join(
                        os.path.dirname(__file__), "..", "..",
                        "datasets", "original", f"{dataset_id.strip()}.csv"
                    )
                    if os.path.isfile(path):
                        df = pd.read_csv(path)
            except Exception:
                pass

        # --- Top-level metrics ---
        st.subheader("Dataset Overview")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Transactions", f"{stats['total_transactions']:,}")
        col2.metric("Columns", stats["columns"])
        col3.metric("Missing Values", stats["missing_values"])
        col4.metric("Dataset Type", stats["dataset_type"].capitalize())

        # Fraud metrics
        if stats.get("fraud_ratio") is not None:
            st.divider()
            col1, col2, col3 = st.columns(3)
            col1.metric("Fraud Ratio", f"{stats['fraud_ratio']:.2%}")
            col2.metric("Normal Transactions", f"{stats.get('normal_count', 0):,}")
            col3.metric("Fraud Transactions", f"{stats.get('fraud_count', 0):,}")

            # Pie chart: Fraud vs Normal
            fig = px.pie(
                names=["Normal", "Fraud"],
                values=[stats.get("normal_count", 0), stats.get("fraud_count", 0)],
                color=["Normal", "Fraud"],
                color_discrete_map={"Normal": "#636EFA", "Fraud": "#EF553B"},
                title="Fraud vs Normal Transactions",
                hole=0.4,
            )
            fig.update_layout(height=350)
            st.plotly_chart(fig, key=None)

        # --- Charts from actual data ---
        if df is not None and not df.empty:
            st.divider()
            st.subheader("Data Visualisations")

            # Transaction amount distribution
            if "amount" in df.columns:
                fig = px.histogram(
                    df, x="amount", nbins=50,
                    title="Transaction Amount Distribution",
                    color_discrete_sequence=["#636EFA"],
                    labels={"amount": "Amount", "count": "Frequency"},
                )
                fig.update_layout(height=400, bargap=0.05)
                st.plotly_chart(fig, key=None)

            # Amount by transaction type (box plot)
            if "amount" in df.columns and "transaction_type" in df.columns:
                fig = px.box(
                    df, x="transaction_type", y="amount",
                    title="Amount by Transaction Type",
                    color="transaction_type",
                    color_discrete_sequence=px.colors.qualitative.Set2,
                )
                fig.update_layout(height=400, showlegend=False)
                st.plotly_chart(fig, key=None)

            # Correlation heatmap
            numeric_df = df.select_dtypes(include=[np.number])
            if len(numeric_df.columns) >= 2:
                corr = numeric_df.corr()
                fig = go.Figure(data=go.Heatmap(
                    z=corr.values,
                    x=corr.columns.tolist(),
                    y=corr.columns.tolist(),
                    colorscale="RdBu_r",
                    zmin=-1, zmax=1,
                    text=np.round(corr.values, 2),
                    texttemplate="%{text}",
                    textfont={"size": 11},
                ))
                fig.update_layout(
                    title="Feature Correlation Heatmap",
                    height=450,
                    xaxis_title="", yaxis_title="",
                )
                st.plotly_chart(fig, key=None)

            # Transactions over time
            if "timestamp" in df.columns:
                try:
                    df["_ts"] = pd.to_datetime(df["timestamp"], errors="coerce")
                    ts_counts = df.groupby(df["_ts"].dt.date).size().reset_index(name="count")
                    ts_counts.columns = ["date", "count"]
                    fig = px.line(
                        ts_counts, x="date", y="count",
                        title="Transactions Over Time",
                        labels={"date": "Date", "count": "Transactions"},
                        color_discrete_sequence=["#00CC96"],
                    )
                    fig.update_layout(height=350)
                    st.plotly_chart(fig, key=None)
                except Exception:
                    pass

        # --- Numeric summary table ---
        st.divider()
        st.subheader("Column Statistics")
        summary = stats.get("summary", {})
        if summary:
            summary_df = pd.DataFrame(summary).T
            st.dataframe(summary_df, width=None)
