# AQI Predictor

Predicting Air Quality Index (AQI) 3 days ahead for Rawalpindi, Pakistan, using a serverless ML pipeline.

## Project Overview
- **Data Source:** OpenWeather Air Pollution API + Open-Meteo (historical weather)
- **Feature Store & Model Registry:** Hopsworks
- **Models Tried:** Ridge Regression, Random Forest, XGBoost
- **Final Model:** Random Forest (trained on 3 years of hourly historical data)

## Project Structure
- `src/feature_pipeline.py` — fetches live data, computes features, saves to Hopsworks
- `src/backfill.py` — pulls historical data for training
- `src/training_pipeline.py` — trains and evaluates the model, saves to Model Registry
- `src/check_data.py`, `check_job.py`, `restart_job.py` — utility scripts for debugging Hopsworks

## Status
Actively in development — Internship project for 10 Pearls.