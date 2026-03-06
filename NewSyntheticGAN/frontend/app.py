import streamlit as st
import pandas as pd
import requests
import json
import plotly.express as px
import plotly.graph_objects as go
import time

# --- Setup & Config ---
st.set_page_config(
    page_title="Synthetic Financial Data & Fraud Detection",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants
API_BASE_URL = "http://127.0.0.1:8000/api"

# Initialize Session State
if "dataset_id" not in st.session_state:
    st.session_state.dataset_id = None
if "synthetic_dataset_id" not in st.session_state:
    st.session_state.synthetic_dataset_id = None

# Custom CSS for aesthetic improvements
st.markdown("""
<style>
    .metric-card {
        background-color: #1E2329;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        margin-bottom: 20px;
    }
    .stProgress .st-bo {
        background-color: #4CAF50;
    }
</style>
""", unsafe_allow_html=True)


# --- Helper Functions ---
def api_request(method, endpoint, **kwargs):
    """Helper to make API requests with error handling."""
    url = f"{API_BASE_URL}{endpoint}"
    try:
        if method == "GET":
            response = requests.get(url, **kwargs)
        elif method == "POST":
            response = requests.post(url, **kwargs)
        else:
            return None, "Invalid method"
            
        response.raise_for_status()
        return response.json(), None
    except requests.exceptions.HTTPError as e:
        error_msg = f"HTTP Error: {e.response.status_code}"
        try:
            error_detail = e.response.json().get('detail', '')
            if error_detail:
                error_msg += f" - {error_detail}"
        except:
            pass
        return None, error_msg
    except Exception as e:
        return None, f"Connection Error: Is the backend running at {API_BASE_URL}? Details: {str(e)}"


# --- Main App Sections ---

def render_upload_analyze():
    st.title("📊 Upload & Analyze Financial Data")
    st.write("Upload a dataset to begin. The system will automatically analyze its statistical properties and scan for anomalies.")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("1. Upload Dataset")
        uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=['csv', 'xlsx'])
        
        if uploaded_file is not None:
            if st.button("Upload & Process", type="primary"):
                with st.spinner("Uploading and processing dataset..."):
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                    result, error = api_request("POST", "/dataset/upload", files=files)
                    
                    if error:
                        st.error(error)
                    else:
                        st.success(f"Successfully uploaded: {result['filename']}")
                        st.session_state.dataset_id = result['id']
                        # Reset synthetic ID when new original is uploaded
                        st.session_state.synthetic_dataset_id = None
                        st.rerun()
                        
    with col2:
        if st.session_state.dataset_id:
            st.subheader("2. Dataset Analysis")
            
            with st.spinner("Fetching analysis..."):
                analysis, error = api_request("GET", f"/dataset/{st.session_state.dataset_id}/analysis")
                
            if error:
                st.error(error)
            elif analysis:
                # Top-level metrics
                mc1, mc2, mc3, mc4 = st.columns(4)
                mc1.metric("Total Rows", f"{analysis['row_count']:,}")
                mc2.metric("Total Columns", analysis['column_count'])
                mc3.metric("Fraud Ratio", f"{analysis['fraud_ratio'] * 100:.2f}%")
                mc4.metric("Duplicates", analysis['duplicate_rows'])
                
                # Suspicious Patterns
                if analysis.get('suspicious_patterns'):
                    st.warning("⚠️ Suspicious Patterns Detected")
                    for pattern in analysis['suspicious_patterns']:
                        st.markdown(f"- {pattern}")
                
                # Amount Distribution & Outliers
                tab1, tab2 = st.tabs(["Amount Distribution", "Outliers Detected"])
                
                with tab1:
                    if 'amount_distribution' in analysis:
                        dist = analysis['amount_distribution']
                        dcol1, dcol2, dcol3 = st.columns(3)
                        dcol1.metric("Mean Amount", f"${dist['mean']:.2f}")
                        dcol2.metric("Median Amount", f"${dist['median']:.2f}")
                        dcol3.metric("Max Amount", f"${dist['max']:.2f}")
                        
                        # Create a simple box plot visualization representation
                        fig = go.Figure()
                        fig.add_trace(go.Box(
                            q1=[dist['q25']], median=[dist['median']],
                            q3=[dist['q75']], lowerfence=[dist['min']],
                            upperfence=[dist['max']],
                            name="Transaction Amounts",
                            orientation='h',
                            marker_color='#3b82f6'
                        ))
                        fig.update_layout(height=250, margin=dict(l=0, r=0, t=30, b=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                        st.plotly_chart(fig, use_container_width=True)
                
                with tab2:
                    st.write(f"Found **{analysis['outlier_count']}** outlier transactions.")
                    if analysis['outlier_count'] > 0 and 'outlier_transactions' in analysis:
                        outliers_df = pd.DataFrame(analysis['outlier_transactions'])
                        st.dataframe(outliers_df, use_container_width=True)
        else:
            st.info("Upload a dataset to see the analysis.")

def render_synthetic_generation():
    st.title("🤖 Generate Synthetic Data")
    
    if not st.session_state.dataset_id:
        st.warning("Please upload an original dataset first (Upload & Analyze tab).")
        return
        
    st.write("Generate privacy-preserving synthetic transactions that maintain the statistical properties of your original data while protecting sensitive real-world information.")
    
    st.markdown("### Configuration")
    with st.form("generation_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            # We don't know the exact row count of the uploaded dataset here without another API call,
            # but we'll default to 1000 for demonstration.
            num_rows = st.number_input("Number of Synthetic Rows to Generate", min_value=100, max_value=50000, value=2000, step=100)
            
        with col2:
            st.write("Class Balancing")
            balance_classes = st.toggle("Automatically balance Fraud vs Normal classes", value=True, 
                                      help="If enabled, the generator will attempt to oversample the minority class (fraud) to reach a 50/50 split, ideal for training ML models.")
            
        submit_btn = st.form_submit_button("Generate Synthetic Dataset", type="primary", use_container_width=True)
        
    if submit_btn:
        with st.status("Initializing AI Generator...", expanded=True) as status:
            st.write("Extracting statistical properties and correlations...")
            
            payload = {
                "dataset_id": st.session_state.dataset_id,
                "num_rows": num_rows,
                "balance_classes": balance_classes
            }
            
            st.write("Training Gaussian Copula Synthesizer... (This may take a moment depending on data size)")
            
            # Record start time
            start_time = time.time()
            
            result, error = api_request("POST", "/synthetic/generate", json=payload)
            
            elapsed = time.time() - start_time
            
            if error:
                status.update(label="Generation Failed", state="error", expanded=True)
                st.error(error)
            else:
                status.update(label=f"Generation Complete ({elapsed:.1f}s)", state="complete", expanded=False)
                st.session_state.synthetic_dataset_id = result['id']
                st.success(f"Successfully generated {result['row_count']:,} synthetic rows!")
                st.balloons()
                
    if st.session_state.synthetic_dataset_id:
        st.success(f"✅ Synthetic Dataset ID **{st.session_state.synthetic_dataset_id}** is ready in memory. Head to the 'Compare Datasets' tab to verify its quality.")


def render_comparison():
    st.title("⚖️ Compare Original vs. Synthetic")
    
    if not st.session_state.dataset_id:
        st.warning("Please upload an original dataset first.")
        return
        
    if not st.session_state.synthetic_dataset_id:
        st.warning("Please generate a synthetic dataset first.")
        return
        
    st.write("Evaluate how well the synthetic data captures the behaviors, edges cases, and correlations of the original data.")
    
    if st.button("Run Detailed Comparison", type="primary"):
        with st.spinner("Running Kolmogorov-Smirnov tests and correlation matrix comparisons..."):
            params = {
                "original_id": st.session_state.dataset_id,
                "synthetic_id": st.session_state.synthetic_dataset_id
            }
            comparison, error = api_request("GET", "/dataset/compare", params=params)
            
            if error:
                st.error(error)
            else:
                st.subheader("Quality Metrics")
                
                # Main Scores
                score_col1, score_col2 = st.columns(2)
                
                with score_col1:
                    col_sim = comparison['column_distribution_similarity']
                    # Define color based on score
                    color = "normal" if col_sim > 0.8 else "inverse" if col_sim < 0.5 else "off"
                    st.metric("Column Distribution Match", f"{col_sim * 100:.1f}%", help="Average KS-Test similarity across all columns.")
                    st.progress(col_sim)
                    
                with score_col2:
                    cor_sim = comparison['correlation_similarity']
                    st.metric("Correlation Similarity Match", f"{cor_sim * 100:.1f}%", help="Based on the Frobenius norm of the difference between original and synthetic correlation matrices.")
                    st.progress(cor_sim)
                
                st.markdown("---")
                
                # Class Balance Comparison
                st.subheader("Target Class Ratio (Fraud)")
                f_comp = comparison['fraud_ratio_comparison']
                
                f_col1, f_col2 = st.columns(2)
                with f_col1:
                    st.metric("Original Fraud Ratio", f"{f_comp['original'] * 100:.2f}%")
                with f_col2:
                    delta = f"{f_comp['synthetic'] - f_comp['original']:.4f}"
                    st.metric("Synthetic Fraud Ratio", f"{f_comp['synthetic'] * 100:.2f}%", delta=delta, delta_color="off")
                    
                if abs(f_comp['synthetic'] - 0.5) < 0.1 and f_comp['original'] < 0.2:
                    st.info("💡 The synthetic data has heavily balanced the fraud class compared to the original dataset. This is excellent for training ML models.")


def render_model_testing():
    st.title("🎯 Fraud Detection Model Testing")
    
    # We need at least one dataset to train on
    available_datasets = {}
    if st.session_state.dataset_id:
        available_datasets["Original Dataset"] = st.session_state.dataset_id
    if st.session_state.synthetic_dataset_id:
        available_datasets["Synthetic Dataset"] = st.session_state.synthetic_dataset_id
        
    if not available_datasets:
        st.warning("Please upload or generate a dataset first.")
        return
        
    st.write("Train Machine Learning models to detect fraudulent transactions and evaluate their performance.")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Experiment Setup")
        
        target_dataset_name = st.radio("Select Training Dataset", options=list(available_datasets.keys()))
        selected_dataset_id = available_datasets[target_dataset_name]
        
        model_type = st.selectbox(
            "Select ML Algorithm",
            options=["logistic_regression", "random_forest", "xgboost"],
            format_func=lambda x: x.replace("_", " ").title()
        )
        
        if st.button("Train & Evaluate Model", type="primary", use_container_width=True):
            with st.spinner(f"Training {model_type.replace('_', ' ').title()} on {target_dataset_name}..."):
                payload = {
                    "dataset_id": selected_dataset_id,
                    "model_type": model_type
                }
                
                result, error = api_request("POST", "/model/train", json=payload)
                
                if error:
                    st.error(error)
                else:
                    st.session_state.last_model_result = result
                    st.session_state.last_model_name = model_type
                    st.session_state.last_dataset_name = target_dataset_name
                    st.rerun()
                    
    with col2:
        st.subheader("Evaluation Results")
        
        if hasattr(st.session_state, 'last_model_result'):
            result = st.session_state.last_model_result
            m_name = st.session_state.last_model_name.replace('_', ' ').title()
            d_name = st.session_state.last_dataset_name
            
            st.success(f"✅ {m_name} trained successfully on {d_name}!")
            
            # Metrics
            mcol1, mcol2, mcol3, mcol4 = st.columns(4)
            mcol1.metric("Accuracy", f"{result['accuracy'] * 100:.1f}%")
            mcol2.metric("Precision", f"{result['precision'] * 100:.1f}%")
            mcol3.metric("Recall", f"{result['recall'] * 100:.1f}%")
            mcol4.metric("F1-Score", f"{result['f1_score'] * 100:.1f}%")
            
            st.markdown("---")
            
            # Confusion Matrix Visualization
            st.write("#### Confusion Matrix")
            cm = result['confusion_matrix']
            
            if len(cm) == 2 and len(cm[0]) == 2:
                # Format for Plotly Heatmap
                z = [[cm[1][0], cm[1][1]], 
                     [cm[0][0], cm[0][1]]]
                
                fig = px.imshow(z, text_auto=True, 
                               labels=dict(x="Predicted Label", y="True Label", color="Count"),
                               x=['Normal (0)', 'Fraud (1)'],
                               y=['Fraud (1)', 'Normal (0)'],
                               color_continuous_scale='Blues')
                
                fig.update_layout(height=400, margin=dict(l=0, r=0, t=30, b=0))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.write(cm) # Fallback if not 2x2
        else:
            st.info("Configure your experiment and click 'Train' to see results.")


# --- Sidebar Navigation ---
st.sidebar.title("💳 Synthetic Data & Fraud Testing")
st.sidebar.markdown("---")

page = st.sidebar.radio("Navigation", [
    "1. Upload & Analyze", 
    "2. Generate Synthetic Data", 
    "3. Compare Datasets", 
    "4. Test Fraud Models"
])

st.sidebar.markdown("---")
st.sidebar.subheader("System Status")

# API Health Check check
try:
    health_res = requests.get(f"http://127.0.0.1:8000/", timeout=2)
    if health_res.status_code == 200:
        st.sidebar.success("Backend API: **Online**")
    else:
        st.sidebar.error("Backend API: **Error**")
except:
    st.sidebar.error("Backend API: **Offline**")

st.sidebar.info(f"Original ID: {st.session_state.dataset_id}\n\nSynthetic ID: {st.session_state.synthetic_dataset_id}")


# --- Routing ---
if page == "1. Upload & Analyze":
    render_upload_analyze()
elif page == "2. Generate Synthetic Data":
    render_synthetic_generation()
elif page == "3. Compare Datasets":
    render_comparison()
elif page == "4. Test Fraud Models":
    render_model_testing()
