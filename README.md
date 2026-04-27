# CBSFIM — Customer Behavior & Strategic Financial Impact Modeling

> **Banking AI platform** — Simulation, Machine Learning, BI, and Executive Dashboard

## Live Demo
🔗 **[Open Dashboard →](https://your-app-url.streamlit.app)**
*(replace this link after Streamlit Cloud deployment)*

---

## What it does

Analyses a portfolio of **50,000 synthetic banking customers** over 24 months:

- **Churn prediction** — XGBoost classifier (AUC ≈ 0.93)
- **Revenue forecasting** — XGBoost regressor
- **Customer Lifetime Value (CLV)** — actuarial 5-year discount model
- **AI recommendations** — rule-based + ML hybrid engine
- **BI Feature Engineering** — 40+ behavioral, financial & composite features
- **Executive KPIs** — Revenue at Risk, Churn Rate, HVAR, Net Profit, Engagement Index
- **Interactive Dashboard** — 7-tab Streamlit + Plotly (white theme)

## Dashboard Tabs

| Tab | Content |
|-----|---------|
| 📊 Executive Summary | KPI cards, risk distribution, segment breakdown |
| 🔍 Customer Risk Explorer | Search by ID, sortable table, CLV vs Churn scatter |
| 💰 Financial Impact | Profitability heatmap, net profit distribution |
| 📈 Revenue Trends | 24-month time series, MoM growth, transaction volume |
| 💡 BI Insights | Engagement analytics, ROI simulator, risk clustering |
| 🤖 AI Recommendations | Action distribution, top-25 priority customers |
| ⚙️ Model Info | Pipeline architecture, simulation parameters |

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Data Simulation | NumPy / Pandas (log-normal + Poisson distributions) |
| Storage | Partitioned Parquet (PyArrow) |
| Churn Model | XGBoost Classifier |
| Revenue Model | XGBoost Regressor |
| CLV Scoring | Actuarial discount model (r=10%, 5yr horizon) |
| Feature Engineering | Pandas + Scikit-learn KMeans |
| Dashboard | Streamlit + Plotly Express & Graph Objects |

## Run Locally

```bash
pip install -r requirements.txt
python pipelines/run_pipeline.py       # ML pipeline (~3 min)
python pipelines/run_bi_pipeline.py   # BI layer  (~1 min)
streamlit run dashboard/app.py        # → http://localhost:8501
```

---
*All data is 100% synthetic — for demonstration purposes only.*
