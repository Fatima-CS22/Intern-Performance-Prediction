# Intern Performance Prediction Model

A machine learning regression project that predicts intern performance scores (0–100) based on completion time, feedback rating and attendance built with Random Forest and XGBoost and deployed as an interactive Streamlit dashboard.

🔗 **Live Demo:** [(https://intern-performance-prediction-72xoychhulacdgzecyvcvf.streamlit.app)]

## Overview

This project predicts an intern's `Performance_Score` using three raw inputs (`Completion_Time`, `Feedback_Rating`, `Attendance`) plus three engineered features derived from them. Both models are trained, evaluated, and compared, and the better-performing model (XGBoost) is used for final predictions. A Streamlit dashboard lets users either upload a CSV for batch predictions or use sliders for a single real-time prediction.

## Dataset
The dataset used in this project was obtained from Kaggle.
The dataset Name: (`intern_dataset_realistic.csv`) contains 10,000 intern records with the following columns:

| Column | Description |
|---|---|
| `Intern_ID` | Unique identifier |
| `Completion_Time` | Time taken to complete tasks (hours) |
| `Feedback_Rating` | Supervisor feedback rating (1–5) |
| `Attendance` | Attendance percentage (0–100) |
| `Performance_Score` | Target variable (0–100) |

Missing values in `Completion_Time`, `Feedback_Rating`, and `Attendance` were imputed using the median (robust to outliers).

## Feature Engineering

Three additional features were derived from the raw columns:

- **Completion_Efficiency** = `Attendance / (Completion_Time + 1)`
- **Overall_Feedback_Attendance** = `Feedback_Rating * Attendance / 100`
- **Feedback_per_Hour** = `Feedback_Rating / (Completion_Time + 1)`

## Models & Results

Two regression models were trained on an 80/20 train-test split (`random_state=42`) and validated with 5-fold cross-validation:

| Model | RMSE | MAE | R² |
|---|---|---|---|
| Random Forest | 9.40 | 7.57 | 0.608 |
| **XGBoost** | **9.20** | **7.39** | **0.625** |

XGBoost performed slightly better and was used for the final batch predictions and category classification.

Interns are classified into four performance categories based on predicted score:

| Category | Score Range |
|---|---|
| 🏆 Excel | 80–100 |
| ✅ Good | 60–79 |
| ⚠️ Average | 45–59 |
| 🔴 Struggle | 0–44 |

## Dashboard Features

The Streamlit app (`app.py`) includes:

- **Model switcher** — toggle between Random Forest and XGBoost, with live R²/RMSE/MAE metrics
- **Batch Analysis tab** — upload a CSV to get predictions, KPI summary, category breakdown chart, score distribution, actual-vs-predicted scatter plot (if ground truth is present), feature importance chart, top/bottom performer tables, and a downloadable results CSV
- **Single Prediction tab** — adjust sliders for a single intern's profile and get an instant score, gauge chart, feature breakdown, and a tailored recommendation

## Project Structure

```
.
├── Task1.ipynb                  # Full training notebook (EDA, feature engineering, training, evaluation)
├── app.py                       # Streamlit dashboard
├── random_forest_model.pkl      # Trained Random Forest model
├── xgboost_model.pkl            # Trained XGBoost model
├── features.pkl                 # List of feature names used by the models
├── model_metrics.pkl            # Saved evaluation metrics for both models
├── requirements.txt             # Python dependencies
└── README.md
```

## Running Locally

```bash
# Clone the repo
git clone https://github.com/Fatima-CS22/Intern-Performance-Prediction.git
cd Intern-Performance-Prediction

# Install dependencies
pip install -r requirements.txt

# Run the dashboard
streamlit run app.py
```

The app will open at `http://localhost:8501`.

## Tech Stack

- **Python**, **pandas**, **NumPy** — data processing
- **scikit-learn** — Random Forest, train-test split, metrics
- **XGBoost** — gradient boosting model
- **Streamlit** — interactive dashboard
- **Plotly** — charts and visualizations

## Author

Fatima Waseem
[LinkedIn](https://linkedin.com/in/fatima-waseem-608604335)
