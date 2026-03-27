# FitLife AI — Consumer Insights Dashboard

## Overview
An interactive Streamlit dashboard analysing 2,000 synthetic Indian consumer survey responses for **FitLife AI** — an AI-powered personalised fitness and nutrition platform targeting the Indian market.

The dashboard covers the full analytical pipeline from raw data through descriptive and diagnostic analysis, aligned to the business strategy of building a personalised, AI-first fitness platform with 20%+ annual growth.

---

## Dashboard Tabs

| Tab | Content |
|-----|---------|
| **Business Overview** | Strategy, product mix, market sizing, target personas |
| **Raw Data** | Full dataset viewer with filters and export |
| **Data Cleaning & Quality** | Missing values, noise, outliers, cleaned dataset |
| **EDA: Demographics** | Age, city tier, income, diet preference distributions |
| **EDA: Fitness & Behaviour** | Goals, activity levels, workout types, motivation, self-efficacy |
| **EDA: Spending & Attitude** | WTP, current spend, barriers, tech adoption, data comfort |
| **Diagnostic Analysis** | Correlations, persona profiles, target variable deep-dive |

---

## Project Structure

```
fitlife_dashboard/
├── app.py                        # Main Streamlit application
├── requirements.txt              # Python dependencies
├── README.md                     # This file
├── data/
│   └── fitlife_ai_survey_dataset.csv   # 2,000 respondent dataset
└── .streamlit/
    └── config.toml               # Theme and server configuration
```

---

## Local Setup

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/fitlife-dashboard.git
cd fitlife-dashboard

# 2. Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the dashboard
streamlit run app.py
```

The dashboard will open at `http://localhost:8501`

---

## Deploying on Streamlit Cloud

1. Push this repository to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click **New app**
4. Select your repository, branch (`main`), and set **Main file path** to `app.py`
5. Click **Deploy** — Streamlit Cloud handles the rest

> No additional configuration needed. The `requirements.txt` and `.streamlit/config.toml` are auto-detected.

---

## Dataset

- **2,000 synthetic respondents** modelled on realistic Indian consumer behaviour
- **48 columns** covering 25 survey questions
- **6 engineered personas**: Aspiring Urban Professional, Budget-Conscious Student, Health-Driven Middle-Ager, Sedentary Skeptic, Homemaker Wellness Seeker, Fitness Enthusiast
- **Deliberate imperfections**: ~6% missing income data, ~4% activity noise, ~2% aspirational WTP outliers
- **Target variable**: `q25_subscription_intent` (Interested / Undecided / Not Interested)

---

## Business Context

FitLife AI is targeting the Indian fitness-tech market — projected to grow at 19.5% CAGR in Asia Pacific. The platform's 7 core offerings are:
1. Adaptive AI Workout Engine
2. Culturally Intelligent Nutrition Coach (Indian diets)
3. AI Recovery & Sleep Optimisation Module
4. Personalised Progress Analytics Dashboard
5. AI Form & Movement Coach (Camera-Based)
6. Corporate Wellness B2B Portal
7. Longevity & Specialty Programs

---

## Tech Stack
- **Frontend**: Streamlit
- **Data**: Pandas, NumPy
- **Visualisation**: Plotly Express & Graph Objects
- **Analysis**: SciPy, Scikit-learn

---

*Built as part of FitLife AI's market research and product strategy initiative — 2026*
