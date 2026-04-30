# ✈️ Flight Price Prediction : Airline Fare Forecasting : MLOps Deployment

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104.1-green?logo=fastapi)
![Streamlit](https://img.shields.io/badge/Streamlit-1.40.0-red?logo=streamlit)
![Docker](https://img.shields.io/badge/Docker-Containerized-blue?logo=docker)
![AWS](https://img.shields.io/badge/AWS-EC2%20%7C%20RDS-orange?logo=amazonaws)
![XGBoost](https://img.shields.io/badge/XGBoost-3.2.0-yellow)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-RDS-blue?logo=postgresql)

## 🚀 Live Demo

**👉 [Click here to try the Live App](http://65.2.39.115:8501)**

---

## 📌 Project Overview

An end-to-end Machine Learning project that predicts flight ticket prices based on route, flight type, booking agency, distance, and travel date. The project covers the complete MLOps lifecycle — from data analysis and model building to containerized cloud deployment on AWS.

**Model Performance:**
| Metric | Train | Test |
|--------|-------|------|
| R² Score | 0.9958 (99.58%) | 0.9958 (99.58%) |
| MAE | $18.72 | $18.70 |
| RMSE | $23.12 | $23.08 |

---

## 🏗️ Architecture

```
User (Browser)
     │
     ▼
┌─────────────────┐
│  Streamlit UI   │  ← Port 8501 (AWS EC2)
│   (Frontend)    │
└────────┬────────┘
         │ HTTP POST /predict
         ▼
┌─────────────────┐
│   FastAPI API   │  ← Port 8000 (AWS EC2)
│   (Backend)     │
└────────┬────────┘
         │
    ┌────┴────┐
    ▼         ▼
┌────────┐  ┌──────────────────┐
│XGBoost │  │  AWS RDS         │
│ Model  │  │  PostgreSQL DB   │
│(joblib)│  │  (Predictions)   │
└────────┘  └──────────────────┘

All services run inside Docker containers on AWS EC2
```

---

## 🛠️ Tech Stack

| Category | Technology |
|----------|-----------|
| ML Model | XGBoost, Scikit-learn |
| Backend API | FastAPI, Uvicorn |
| Frontend | Streamlit |
| Database | PostgreSQL (AWS RDS) |
| Containerization | Docker, Docker Compose |
| Cloud | AWS EC2, AWS RDS |
| Image Registry | Docker Hub |
| Language | Python 3.11 |

---

## 📁 Project Structure

```
FLIGHT_PRICE_PREDICTION/
├── main.py                        # FastAPI backend — loads model, serves /predict
├── app.py                         # Streamlit frontend — user interface
├── postgres.py                    # Database connection test
├── Dockerfile.fastapi             # Docker instructions for FastAPI
├── Dockerfile.streamlit           # Docker instructions for Streamlit
├── docker-compose.yml             # Orchestrates both containers
├── requirements.txt               # Python dependencies
├── .env.example                   # Environment variables template
├── xgb_flight_price_model.joblib  # Trained XGBoost model
├── standard_scaler.joblib         # Fitted StandardScaler
├── feature_columns.joblib         # Feature column names
└── README.md
```

---

## 📊 Dataset

- **Source:** Brazilian flight price dataset
- **Rows:** 271,888
- **Columns:** 10
- **Features Used:** `from`, `to`, `flightType`, `agency`, `time`, `distance`, `day`, `month`, `is_weekend`, `speed`

---

## ⚙️ Feature Engineering

| Feature | Description |
|---------|-------------|
| `day` | Day of week extracted from date (0=Monday) |
| `month` | Month extracted from date |
| `is_weekend` | 1 if Saturday/Sunday, else 0 |
| `speed` | distance / time (km/h) |

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health check |
| POST | `/predict` | Predict flight price |
| GET | `/predictions` | Get stored predictions |

**Sample Request:**
```json
{
  "from_city": "Rio de Janeiro (RJ)",
  "to_city": "Brasilia (DF)",
  "flight_type": "economic",
  "agency": "CloudFy",
  "time": 2.0,
  "distance": 1000.0,
  "date": "2026-05-01"
}
```

**Sample Response:**
```json
{
  "predicted_price": 702.25,
  "from_city": "Rio de Janeiro (RJ)",
  "to_city": "Brasilia (DF)",
  "flight_type": "economic",
  "agency": "CloudFy",
  "date": "2026-05-01"
}
```

---

## 🐳 Run Locally with Docker

**1. Clone the repository:**
```bash
git clone https://github.com/GOWTHAMVANTAKULA/FLIGHT_PRICE_PREDICTION_ML_END_TO_END_PROJECT.git
cd FLIGHT_PRICE_PREDICTION_ML_END_TO_END_PROJECT
```

**2. Create `.env` file:**
```
DATABASE_URL=postgresql://username:password@your-rds-endpoint:5432/dbname
API_URL=http://localhost:8000
```

**3. Build and run:**
```bash
docker-compose up --build
```

**4. Open in browser:**
- Streamlit App → http://localhost:8501
- FastAPI Docs → http://localhost:8000/docs

---

## 💻 Run Locally without Docker

**1. Create virtual environment:**
```bash
python -m venv venv
venv\Scripts\activate  # Windows
```

**2. Install dependencies:**
```bash
pip install -r requirements.txt
```

**3. Run FastAPI (Terminal 1):**
```bash
uvicorn main:app --reload
```

**4. Run Streamlit (Terminal 2):**
```bash
streamlit run app.py
```

---

## ☁️ AWS Deployment

The app is deployed on **AWS EC2 (t3.micro)** with the following setup:

- **EC2:** Ubuntu 22.04 LTS — hosts Docker containers
- **RDS:** PostgreSQL — stores all predictions
- **Docker Hub:** [`gowtham2733/flight-fastapi`](https://hub.docker.com/u/gowtham2733) — image registry
- **Security Groups:** Ports 22, 8000, 8501 open

**Deployment steps:**
```bash
# On EC2
docker pull gowtham2733/flight-fastapi:latest
docker pull gowtham2733/flight-streamlit:latest
docker compose up -d
```





## 👤 Author

**Vantakula Gowtham Naidu**

[![GitHub](https://img.shields.io/badge/GitHub-GOWTHAMVANTAKULA-black?logo=github)](https://github.com/GOWTHAMVANTAKULA)
[![Docker Hub](https://img.shields.io/badge/Docker%20Hub-gowtham2733-blue?logo=docker)](https://hub.docker.com/u/gowtham2733)




