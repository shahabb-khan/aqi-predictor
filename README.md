# AQI Predictor — Rawalpindi, Pakistan

A fully serverless machine learning system that predicts Air Quality Index (AQI) for Rawalpindi, Pakistan, up to 3 days in advance.

**🔴 Live Dashboard:** https://aqi-predictor-72d8k3vx5ndax4obtuddf4.streamlit.app/

10Pearls Data Science Internship project by Shahab Uddin, mentored by Hafsa Imtiaz and Umema Ashar.

---

## What It Does

Rawalpindi's air quality is unhealthy most of the year (mean AQI ≈ 165 across 3 years of hourly data). This project builds an automated forecasting system so residents can plan ahead:

- Fetches live pollution + weather data every hour
- Retrains 3 separate machine learning models daily (Day 1, Day 2, Day 3 forecasts)
- Serves predictions through a live, public dashboard

## Results

| Model | RMSE | MAE | R² |
|---|---|---|---|
| Day 1 (24h ahead) | 19.06 | 15.21 | **0.710** |
| Day 2 (48h ahead) | 35.13 | 28.32 | 0.000 |
| Day 3 (72h ahead) | 37.59 | 31.71 | -0.174 |

The Day 1 model meets the project's target accuracy. Day 2/3 accuracy drops off — an expected characteristic of forecasting further into the future, not a defect (see full report for details).

## Tech Stack

- **Modeling:** Python, Scikit-learn (Random Forest)
- **Feature Store / Model Registry:** Hopsworks
- **Automation:** GitHub Actions (hourly feature pipeline, daily training pipeline)
- **Dashboard:** Streamlit, deployed on Streamlit Community Cloud
- **Data sources:** OpenWeather Air Pollution API, Open-Meteo (weather)
- **Explainability:** SHAP

## Project Structure

```
aqi-predictor/
├── .github/workflows/       # CI/CD: hourly + daily automation
├── src/
│   ├── feature_pipeline.py  # Hourly data collection
│   ├── training_pipeline.py # Daily training of 3 day-specific models
│   └── ...                  # utilities
├── app.py                   # Streamlit dashboard
├── reports/
│   └── AQI_Predictor_Report.docx   # Full write-up: methodology, EDA, SHAP, results
├── requirements.txt
└── README.md
```

## Full Report

See [`reports/AQI_Predictor_Report.docx`](./reports/AQI_Predictor_Report.docx) for the complete write-up, including exploratory data analysis, model explainability (SHAP), and detailed discussion of results.

## Running Locally

```bash
python -m venv venv
venv\Scripts\activate      # Windows
pip install -r requirements.txt
streamlit run app.py
```

Requires a `.env` file with `HOPSWORKS_API_KEY` and `OPENWEATHER_API_KEY`.