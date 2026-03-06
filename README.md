# NewSyntheticGAN: Synthetic Financial Data & Fraud Detection Platform

An AI-powered, full-stack platform designed to generate privacy-preserving synthetic financial transaction data and evaluate machine learning models for fraud detection. 

The system leverages state-of-the-art synthetic data generation techniques to maintain the statistical properties and correlations of your original data while protecting sensitive real-world information, making it ideal for training robust fraud detection models.

---

## 🌟 Key Features

1. **Upload & Analyze Financial Data**
   - Upload financial datasets (CSV/Excel).
   - Automatically extract statistical properties, analyze data distribution, and detect outliers or suspicious patterns.
   
2. **AI-Powered Synthetic Data Generation**
   - Uses **SDV (Gaussian Copula Synthesizer)** to generate highly realistic synthetic transactions.
   - **Automatic Class Balancing**: Option to automatically oversample minority classes (e.g., fraud) to achieve a 50/50 split, drastically improving the performance of Machine Learning models trained on the data.
   
3. **Compare Original vs. Synthetic Datasets**
   - **Kolmogorov-Smirnov (KS) Tests**: Evaluates column distribution similarity.
   - **Correlation Matrix Similarity**: Compares the relationships between variables in both datasets.
   - **Fraud Ratio Comparison**: Tracks how the target class ratio shifts between the original and synthetic data.
   
4. **Fraud Detection Model Testing**
   - Train and evaluate ML models directly from the dashboard on either the original or the generated synthetic dataset.
   - Supported Algorithms: **Logistic Regression**, **Random Forest**, **XGBoost**.
   - View detailed evaluation metrics including Accuracy, Precision, Recall, F1-Score, and an interactive Confusion Matrix.

---

## 🛠️ Technology Stack

### Backend
* **Framework**: FastAPI
* **Database**: PostgreSQL (via `asyncpg` and SQLAlchemy async)
* **Synthetic Data Generation**: SDV (Synthetic Data Vault), Faker
* **Machine Learning & Stats**: Scikit-Learn, XGBoost, SciPy
* **Data Processing**: Pandas, NumPy

### Frontend
* **Framework**: Streamlit
* **Visualizations**: Plotly
* **Data Handling**: Pandas, Requests

---

## 🚀 Getting Started

### Prerequisites
* Python 3.10+
* PostgreSQL database

### 1. Backend Setup

Navigate to the `backend` directory and install the dependencies:

```bash
cd NewSyntheticGAN/backend
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
pip install -r requirements.txt
```

Create a `.env` file in the `backend` directory with your database credentials (refer to `.env.example` if available), for example:
```env
DATABASE_URL=postgresql+asyncpg://user:password@localhost:5432/synthetic_db
```

Start the FastAPI server:
```bash
uvicorn app.main:app --reload
```
*The backend API will be available at `http://127.0.0.1:8000`*

### 2. Frontend Setup

Open a new terminal, navigate to the `frontend` directory, and install its dependencies:

```bash
cd NewSyntheticGAN/frontend
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
pip install -r requirements.txt
```

Start the Streamlit dashboard:
```bash
streamlit run app.py
```
*The dashboard will automatically open in your browser at `http://localhost:8501`.*

---

## 📂 Project Structure

```text
NewSyntheticGAN/
├── backend/
│   ├── app/
│   │   ├── main.py            # FastAPI application entry point
│   │   ├── config.py          # Application settings and configurations
│   │   ├── database.py        # SQLAlchemy async database setup
│   │   ├── models/            # Database models / Schema
│   │   ├── routers/           # API Endpoints (dataset, synthetic, model)
│   │   ├── services/          # Business logic (SDV synthesis, ML training)
│   │   └── utils/             # Helper functions
│   └── requirements.txt       # Backend dependencies
└── frontend/
    ├── app.py                 # Streamlit dashboard application
    └── requirements.txt       # Frontend dependencies
```

---

## 🛡️ License & Disclaimer
This project was developed as part of a hackathon (CVMU 2026). The generated synthetic data is for testing and development purposes only and should not be used as genuine financial advice or real-world transaction data.
