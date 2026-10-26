# Churn Prediction & Retention Economics Tool

A production-grade churn prediction system that translates ML model outputs into **business-facing economics** — quantifying revenue at risk, customer lifetime value, and optimal retention-budget allocation by segment.

## 🏗 Architecture

```
churn-retention-economics/
├── config.py                # Central configuration (paths, hyperparams, economics)
├── data_pipeline.py         # Load → clean → feature-engineer → encode → split
├── churn_model.py           # XGBoost training, evaluation, scoring, persistence
├── retention_economics.py   # CLV, revenue-at-risk, segment ROI modelling
├── train.py                 # One-shot training pipeline
├── dashboard.py             # Streamlit CLV-driven dashboard
├── data/                    # Drop IBM Telco CSV here
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
└── artifacts/               # Auto-generated model + economics artefacts
    ├── xgb_churn_model.joblib
    ├── model_metrics.json
    ├── feature_names.json
    ├── feature_importance.csv
    ├── customer_economics.csv
    └── segment_economics.csv
```

## 🚀 Quick Start

### 1. Place your data
Download the [IBM Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) and save it as:
```
data/WA_Fn-UseC_-Telco-Customer-Churn.csv
```

### 2. Train the model
```bash
python train.py
```
This will:
- Load and feature-engineer the data
- Train an XGBoost churn classifier (handles class imbalance)
- Evaluate on a held-out test set
- Compute CLV and revenue-at-risk for every customer
- Build segment-level retention economics
- Save all artifacts to `artifacts/`

### 3. Launch the dashboard
```bash
streamlit run dashboard.py
```

## 📊 Dashboard Views

| View | What it shows |
|------|---------------|
| **Executive Summary** | KPI cards, risk-segment snapshot, revenue-at-risk bar chart, CLV donut, churn distribution, top churn drivers |
| **Retention ROI** | Recommended budget allocation, per-segment ROI, economics detail table, revenue-recovery waterfall |
| **Customer Deep-Dive** | Filterable CLV vs churn scatter, sortable customer table with economic overlays |
| **Model Performance** | Confusion matrix, feature importance, precision/recall/AUC metrics |

## 💡 Key Concepts

### Customer Lifetime Value (CLV)
Discounted present-value of future margin over a 36-month horizon:

$$CLV = \sum_{t=1}^{36} \frac{Margin \times (1 - P_{churn})^t}{(1 + r/12)^t}$$

### Revenue-at-Risk
$$R_{risk} = CLV \times P_{churn}$$

### Retention ROI
For each segment, we model an intervention with:
- **Cost/customer** — segment-specific outreach cost
- **Lift** — estimated increase in retention probability

$$ROI = \frac{Lift \times R_{risk} - Cost}{Cost}$$

## ⚙️ Configuration

All tuneable parameters live in [`config.py`](config.py):
- Feature lists
- CLV economics (discount rate, margin rate)
- Retention costs & intervention lifts per segment
- XGBoost hyperparameters
- Risk-segment bin boundaries

## 🛠 Tech Stack
- **ML:** XGBoost, Scikit-learn
- **Data:** Pandas, NumPy
- **Visualisation:** Plotly, Streamlit
- **Explainability:** SHAP (available for analysis)
