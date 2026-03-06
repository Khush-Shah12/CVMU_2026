"""
frontend/app.py
===============
Main Streamlit dashboard for the Financial Synthetic Data Generator.

Provides sidebar navigation across 5 features:
  1. Upload Dataset
  2. Generate Synthetic Data
  3. Validate Synthetic Data
  4. Analytics Dashboard
  5. Download Dataset

Run with:
    streamlit run frontend/app.py
"""

import streamlit as st
import sys
import os

# Add frontend directory to path so page imports work
sys.path.insert(0, os.path.dirname(__file__))

from pages import upload_dataset, generate_data, validate_data, analytics_dashboard
from utils.api_client import download_synthetic

# --------------------------------------------------------------------------
# Page config
# --------------------------------------------------------------------------

st.set_page_config(
    page_title="AI Financial Data Generator",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------
# Custom CSS for premium look
# --------------------------------------------------------------------------

st.markdown("""
<style>
    /* Main title styling */
    .main-title {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 2.2rem;
        font-weight: 800;
        margin-bottom: 0;
        line-height: 1.3;
    }
    .sub-title {
        font-size: 1.1rem;
        margin-top: -10px;
        opacity: 0.75;
    }

    /* Sidebar - subtle gradient that works with both themes */
    [data-testid="stSidebar"] > div:first-child {
        background: linear-gradient(
            180deg,
            rgba(102, 126, 234, 0.08) 0%,
            rgba(118, 75, 162, 0.08) 100%
        );
    }
    [data-testid="stSidebar"] .stRadio label {
        font-size: 1.05rem;
        padding: 6px 0;
    }

    /* Metric cards - theme-aware */
    [data-testid="stMetric"] {
        border: 1px solid rgba(128, 128, 128, 0.2);
        border-radius: 12px;
        padding: 16px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    }
    [data-testid="stMetricLabel"] p {
        font-weight: 600;
    }

    /* Primary buttons */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
        border: none;
        font-weight: 600;
        font-size: 1.05rem;
        padding: 12px 24px;
        border-radius: 8px;
    }
    .stButton > button[kind="primary"]:hover {
        background: linear-gradient(135deg, #5a6fd6 0%, #6a4296 100%);
        color: white !important;
    }

    /* Divider */
    hr {
        opacity: 0.3;
    }

    /* Expander headers */
    .streamlit-expanderHeader {
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------

with st.sidebar:
    st.markdown("### 🏦 Navigation")
    st.divider()

    page = st.radio(
        "Go to",
        [
            "📤 Upload Dataset",
            "⚙️ Generate Synthetic Data",
            "✅ Validate Synthetic Data",
            "📊 Analytics Dashboard",
            "📥 Download Dataset",
        ],
        label_visibility="collapsed",
    )

    st.divider()

    # Show last dataset ID if available
    if "last_dataset_id" in st.session_state:
        st.caption(f"**Last Dataset ID:**")
        st.code(st.session_state["last_dataset_id"])

    st.divider()
    st.caption("v2.0 | Powered by VAE + Isolation Forest")

# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------

st.markdown('<p class="main-title">AI-Powered Financial Synthetic Data Generator</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-title">'
    'Generate privacy-safe synthetic financial transaction datasets '
    'for fraud detection model training.'
    '</p>',
    unsafe_allow_html=True,
)
st.divider()

# --------------------------------------------------------------------------
# Page routing
# --------------------------------------------------------------------------

if page == "📤 Upload Dataset":
    upload_dataset.render()

elif page == "⚙️ Generate Synthetic Data":
    generate_data.render()

elif page == "✅ Validate Synthetic Data":
    validate_data.render()

elif page == "📊 Analytics Dashboard":
    analytics_dashboard.render()

elif page == "📥 Download Dataset":
    st.header("Download Synthetic Dataset")
    st.markdown("Enter the Dataset ID to download the generated synthetic CSV file.")

    default_id = st.session_state.get("last_dataset_id", "")
    dataset_id = st.text_input(
        "Dataset ID",
        value=default_id,
        placeholder="e.g. ds_ebcca060",
    )

    if st.button("Download CSV", type="primary", use_container_width=True):
        if not dataset_id.strip():
            st.error("Please enter a Dataset ID.")
        else:
            with st.spinner("Preparing download..."):
                try:
                    csv_bytes = download_synthetic(dataset_id.strip())
                    st.success(f"Dataset ready! Size: {len(csv_bytes):,} bytes")

                    st.download_button(
                        label="💾 Save CSV File",
                        data=csv_bytes,
                        file_name=f"{dataset_id.strip()}_synthetic.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )
                except Exception as e:
                    st.error(f"Download failed: {e}")
