"""
frontend/pages/generate_data.py
===============================
Page 2: Generate synthetic financial data from an uploaded dataset.
"""

import streamlit as st
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.api_client import generate_data


def render():
    st.header("Generate Synthetic Data")
    st.markdown(
        "Provide the **Dataset ID** from a previous upload, configure generation "
        "parameters, and create a new synthetic dataset using the AI generator."
    )

    # Auto-fill from session state if available
    default_id = st.session_state.get("last_dataset_id", "")

    dataset_id = st.text_input(
        "Dataset ID",
        value=default_id,
        placeholder="e.g. ds_ebcca060",
        help="ID returned from the Upload page",
    )

    col1, col2 = st.columns(2)
    with col1:
        num_rows = st.number_input(
            "Number of synthetic rows",
            min_value=100,
            max_value=100_000,
            value=1000,
            step=100,
            help="Leave blank or set to match original dataset size",
        )
    with col2:
        fraud_ratio = st.slider(
            "Fraud ratio",
            min_value=0.0,
            max_value=0.30,
            value=0.05,
            step=0.01,
            format="%.2f",
            help="Fraction of transactions to inject as fraud (0-30%)",
        )

    st.divider()

    if st.button("Generate Synthetic Data", type="primary", use_container_width=True):
        if not dataset_id.strip():
            st.error("Please enter a Dataset ID.")
            return

        with st.spinner("Generating synthetic data... This may take 30-60 seconds."):
            try:
                result = generate_data(
                    dataset_id=dataset_id.strip(),
                    num_rows=int(num_rows),
                    fraud_ratio=fraud_ratio,
                )

                st.success("Synthetic data generated successfully!")
                st.session_state["last_dataset_id"] = dataset_id.strip()

                col1, col2, col3 = st.columns(3)
                col1.metric("Dataset ID", result["dataset_id"])
                col2.metric("Synthetic Rows", f"{result['synthetic_rows']:,}")
                col3.metric("Fraud Rows", f"{result['fraud_rows']:,}")

                st.info(f"**Status:** {result['status']}")
                st.caption("You can now validate this dataset or download it.")

            except Exception as e:
                st.error(f"Generation failed: {e}")
