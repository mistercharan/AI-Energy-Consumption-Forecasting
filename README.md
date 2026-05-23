# ⚡ AI-Powered Energy Consumption Forecasting System

> Forecasting hourly electricity demand using Machine Learning to support Smart Cities and Climate Tech.

---

## 📌 Overview

This project builds an end-to-end AI pipeline that predicts hourly energy consumption using historical smart meter data. It simulates how energy companies like **Siemens, Schneider Electric, and Tata Power** use machine learning in production systems.

---

## 🔴 Problem Statement

| Problem | Solution |
|---|---|
| Unpredictable energy demand causes blackouts | AI forecasts demand 48 hours in advance |
| Buildings waste electricity during peak hours | Model identifies wasteful usage patterns |
| Manual monitoring is slow and error-prone | Automated real-time predictions |
| Carbon emissions from over-generation | Aligns supply with actual demand |

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.10 |
| ML Model | XGBoost Regressor |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Dataset | UCI Household Power Consumption |

---

## 📁 Project Structure

```
AI-Energy-Forecasting/
├── src/
│   ├── preprocess.py     ← Data cleaning
│   ├── features.py       ← Feature engineering
│   ├── train.py          ← Model training
│   ├── predict.py        ← Forecasting
│   └── visualize.py      ← Chart generation
├── outputs/
│   ├── graphs/           ← 6 visualizations
│   ├── predictions.csv   ← Model predictions
│   └── metrics.txt       ← Evaluation results
├── main.py               ← Full pipeline runner
└── requirements.txt
```

---

## 📊 Model Results

| Metric | Value |
|---|---|
| R² Score | 0.59 |
| RMSE | 0.46 kWh |
| MAE | 0.32 kWh |
| Forecast Horizon | 48 hours |
| Training Samples | 27,536 |
| Dataset Size | 2,075,259 readings |

---

## 🔧 Feature Engineering

- **Temporal:** Hour, day, month, weekday, season, quarter
- **Lag features:** 1h, 6h, 12h, 24h, 48h, 168h lookback
- **Rolling stats:** 3h, 6h, 24h, 7-day mean and std
- **Domain:** Peak-hour flag, weekend flag

---

## 📈 Output Visualizations

- Actual vs Predicted (last 7 days)
- Energy Heatmap (hour vs day of week)
- 48-hour Forecast Timeline
- Feature Importance Chart
- Monthly Energy Trend
- Error Distribution Analysis

---

## ▶️ How to Run

```bash
pip install -r requirements.txt
python main.py
```

---

## 🏭 Industry Relevance

Companies hiring for this skill set:

**Product-based:** Google, Siemens, Schneider Electric, Honeywell, GE

**Service-based:** TCS, Infosys, Wipro, Accenture, L&T Technology Services

**Energy sector:** Tata Power, ENGIE, National Grid, Shell Energy

---

## 🎓 Learning Outcomes

- End-to-end ML pipeline for time-series regression
- XGBoost with hyperparameter tuning
- Temporal feature engineering with lag and rolling features
- Industry-grade modular Python code structure
- Energy domain knowledge for smart grid applications

---

## 👤 Author

**Charan**
B.E. Electrical and Electronics Engineering
KCG College of Technology, Karapakkam

---

## 📄 License

MIT License — Free to use for educational purposes.
