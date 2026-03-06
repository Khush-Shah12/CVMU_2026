"""
frontend/pages/upload_dataset.py
================================
Page 1: Upload a CSV financial dataset to the backend.
"""

import streamlit as st
import pandas as pd
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.api_client import upload_dataset


def render():
    st.header("Upload Dataset")
    st.markdown(
        "Upload a CSV file containing financial transaction data. "
        "The system will store it and return a **Dataset ID** for further operations."
    )

    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=["csv"],
        help="Maximum file size: 100 MB",
    )

    if uploaded_file is not None:
        # Preview the dataset
        try:
            df = pd.read_csv(uploaded_file)
            st.subheader("Dataset Preview")
            st.dataframe(df.head(20), width=None)

            col1, col2, col3 = st.columns(3)
            col1.metric("Rows", f"{len(df):,}")
            col2.metric("Columns", len(df.columns))
            col3.metric("Missing Values", int(df.isnull().sum().sum()))

            # Reset file pointer for upload
            uploaded_file.seek(0)
        except Exception as e:
            st.error(f"Could not preview file: {e}")

        # Upload button
        if st.button("Upload to Platform", type="primary", use_container_width=True):
            with st.spinner("Uploading dataset..."):
                try:
                    uploaded_file.seek(0)
                    result = upload_dataset(uploaded_file.read(), uploaded_file.name)

                    st.success(f"Dataset uploaded successfully!")

                    # Store dataset_id in session state
                    st.session_state["last_dataset_id"] = result["dataset_id"]

                    # Display results
                    col1, col2 = st.columns(2)
                    col1.info(f"**Dataset ID:** `{result['dataset_id']}`")
                    col2.info(f"**Rows:** {result['rows']}  |  **Columns:** {result['columns']}")

                    st.caption("Save the Dataset ID above — you'll need it for generation and validation.")

                except Exception as e:
                    st.error(f"Upload failed: {e}")
