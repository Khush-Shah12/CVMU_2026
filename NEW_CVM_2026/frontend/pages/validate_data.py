"""
frontend/pages/validate_data.py
===============================
Page 3: Validate synthetic data quality against the original dataset.
"""

import streamlit as st
import plotly.graph_objects as go
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.api_client import validate_data


def _quality_color(value: float, max_val: float = 10.0) -> str:
    """Return a color based on quality score."""
    ratio = value / max_val
    if ratio >= 0.8:
        return "green"
    elif ratio >= 0.5:
        return "orange"
    return "red"


def _match_emoji(label: str) -> str:
    """Return an emoji for match labels."""
    return {
        "excellent": "🟢",
        "good": "🟡",
        "fair": "🟠",
        "poor": "🔴",
    }.get(label, "⚪")


def render():
    st.header("Validate Synthetic Data")
    st.markdown(
        "Compare the generated synthetic dataset against the original. "
        "The AI validator checks statistical distributions, correlations, "
        "fraud patterns, and logical consistency."
    )

    default_id = st.session_state.get("last_dataset_id", "")
    dataset_id = st.text_input(
        "Dataset ID",
        value=default_id,
        placeholder="e.g. ds_ebcca060",
    )

    if st.button("Validate Data", type="primary", use_container_width=True):
        if not dataset_id.strip():
            st.error("Please enter a Dataset ID.")
            return

        with st.spinner("Running validation... This may take 30-60 seconds."):
            try:
                result = validate_data(dataset_id.strip())
            except Exception as e:
                st.error(f"Validation failed: {e}")
                return

        st.success("Validation complete!")

        # --- Top-level metrics ---
        st.subheader("Quality Metrics")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Similarity Score", f"{result['similarity_score']:.2f}")
        col2.metric("Quality Score", f"{result['quality_score']}/10")
        col3.metric("Realism Score", f"{result['realism_score']}/100")
        col4.metric("Anomaly Score", f"{result['anomaly_score']:.4f}")

        col1, col2 = st.columns(2)
        with col1:
            fraud_match = result["fraud_ratio_match"]
            st.info(f"**Fraud Ratio Match:** {_match_emoji(fraud_match)} {fraud_match.capitalize()}")
        with col2:
            corr_match = result["correlation_match"]
            st.info(f"**Correlation Match:** {_match_emoji(corr_match)} {corr_match.capitalize()}")

        if result.get("fraud_patterns_detected"):
            st.warning("Fraud patterns detected in the synthetic dataset.")
        else:
            st.success("No suspicious fraud patterns detected.")

        st.divider()

        # --- Gauge charts ---
        st.subheader("Score Gauges")
        col1, col2, col3 = st.columns(3)

        with col1:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=result["similarity_score"] * 100,
                title={"text": "Similarity"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#636EFA"},
                    "steps": [
                        {"range": [0, 40], "color": "#FECDD3"},
                        {"range": [40, 70], "color": "#FEF3C7"},
                        {"range": [70, 100], "color": "#D1FAE5"},
                    ],
                },
            ))
            fig.update_layout(height=250, margin=dict(t=50, b=0, l=30, r=30))
            st.plotly_chart(fig, key=None)

        with col2:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=result["quality_score"],
                title={"text": "Quality"},
                gauge={
                    "axis": {"range": [0, 10]},
                    "bar": {"color": "#EF553B"},
                    "steps": [
                        {"range": [0, 4], "color": "#FECDD3"},
                        {"range": [4, 7], "color": "#FEF3C7"},
                        {"range": [7, 10], "color": "#D1FAE5"},
                    ],
                },
            ))
            fig.update_layout(height=250, margin=dict(t=50, b=0, l=30, r=30))
            st.plotly_chart(fig, key=None)

        with col3:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=result["realism_score"],
                title={"text": "Realism"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#00CC96"},
                    "steps": [
                        {"range": [0, 40], "color": "#FECDD3"},
                        {"range": [40, 70], "color": "#FEF3C7"},
                        {"range": [70, 100], "color": "#D1FAE5"},
                    ],
                },
            ))
            fig.update_layout(height=250, margin=dict(t=50, b=0, l=30, r=30))
            st.plotly_chart(fig, key=None)

        # --- Detailed report ---
        st.subheader("Detailed Report")
        report = result.get("detailed_report", {})

        with st.expander("Statistical Validation", expanded=False):
            stat = report.get("ai_validation", {}).get("statistical", {})
            if stat:
                cols = st.columns(4)
                cols[0].metric("Mean Amount", f"${stat.get('mean_amount', 0):,.2f}")
                cols[1].metric("Std Amount", f"${stat.get('std_amount', 0):,.2f}")
                cols[2].metric("Skewness", f"{stat.get('skewness', 0):.4f}")
                cols[3].metric("Kurtosis", f"{stat.get('kurtosis', 0):.4f}")

        with st.expander("Distribution Similarity", expanded=False):
            dist = report.get("distribution_similarity", {})
            per_col = dist.get("per_column", {})
            if per_col:
                for col_name, score in per_col.items():
                    st.progress(min(score, 1.0), text=f"{col_name}: {score:.4f}")
                st.metric("Average Similarity", f"{dist.get('average', 0):.4f}")

        with st.expander("Fraud Comparison", expanded=False):
            fraud = report.get("fraud_comparison", {})
            if fraud:
                col1, col2 = st.columns(2)
                col1.metric("Original Fraud Ratio", f"{fraud.get('original_ratio', 0):.4f}")
                col2.metric("Synthetic Fraud Ratio", f"{fraud.get('synthetic_ratio', 0):.4f}")

        with st.expander("Raw JSON Response", expanded=False):
            st.json(result)
