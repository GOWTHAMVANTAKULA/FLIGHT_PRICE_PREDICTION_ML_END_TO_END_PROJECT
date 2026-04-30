from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import numpy as np
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from datetime import datetime
import os

# -------------------------------
# Load environment variables
# -------------------------------
load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")

# -------------------------------
# PostgreSQL Connection
# -------------------------------
engine = create_engine(DATABASE_URL + "?sslmode=require")

# -------------------------------
# Load Model Artifacts
# -------------------------------
model = joblib.load("xgb_flight_price_model.joblib")
scaler = joblib.load("standard_scaler.joblib")
feature_columns = joblib.load("feature_columns.joblib")

# -------------------------------
# FastAPI App
# -------------------------------
app = FastAPI(title="Flight Price Prediction API", version="1.0.0")

# -------------------------------
# Request Schema
# -------------------------------
class FlightInput(BaseModel):
    from_city: str
    to_city: str
    flight_type: str
    agency: str
    time: float
    distance: float
    date: str  # format: YYYY-MM-DD

# -------------------------------
# Create table on startup
# -------------------------------
@app.on_event("startup")
def create_table():
    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS flight_price_predictions (
                id SERIAL PRIMARY KEY,
                from_city VARCHAR(100),
                to_city VARCHAR(100),
                flight_type VARCHAR(50),
                agency VARCHAR(50),
                time FLOAT,
                distance FLOAT,
                date DATE,
                predicted_price FLOAT,
                created_at TIMESTAMP DEFAULT NOW()
            )
        """))

# -------------------------------
# Health check endpoint
# -------------------------------
@app.get("/")
def root():
    return {"status": "Flight Price Prediction API is running!"}

# -------------------------------
# Predict endpoint
# -------------------------------
@app.post("/predict")
def predict(input: FlightInput):

    if input.from_city == input.to_city:
        raise HTTPException(status_code=400, detail="from_city and to_city cannot be the same.")

    # Prepare data
    data = pd.DataFrame({
        "from": [input.from_city],
        "to": [input.to_city],
        "flightType": [input.flight_type],
        "agency": [input.agency],
        "time": [input.time],
        "distance": [input.distance],
        "date": [input.date]
    })

    # Feature engineering
    data["date"] = pd.to_datetime(data["date"])
    data["day"] = data["date"].dt.dayofweek
    data["month"] = data["date"].dt.month
    data["is_weekend"] = data["day"].apply(lambda x: 1 if x >= 5 else 0)
    data["speed"] = data["distance"] / data["time"]
    data.drop(columns=["date"], inplace=True)

    # Encoding
    data = pd.get_dummies(data)
    data = data.reindex(columns=feature_columns, fill_value=0)

    # Scaling
    num_cols = ['time', 'distance', 'day', 'month', 'is_weekend', 'speed']
    data[num_cols] = scaler.transform(data[num_cols])

    # Predict
    prediction = float(model.predict(data)[0])

    # Save to PostgreSQL
    try:
        db_data = pd.DataFrame({
            "from_city": [input.from_city],
            "to_city": [input.to_city],
            "flight_type": [input.flight_type],
            "agency": [input.agency],
            "time": [input.time],
            "distance": [input.distance],
            "date": [input.date],
            "predicted_price": [prediction],
            "created_at": [datetime.now()]
        })
        with engine.begin() as conn:
            db_data.to_sql(
                "flight_price_predictions",
                conn,
                schema="public",
                if_exists="append",
                index=False
            )
    except Exception as e:
        print(f"DB save failed: {e}")

    return {
        "predicted_price": round(prediction, 2),
        "from_city": input.from_city,
        "to_city": input.to_city,
        "flight_type": input.flight_type,
        "agency": input.agency,
        "date": input.date
    }

# -------------------------------
# Get stored predictions
# -------------------------------
@app.get("/predictions")
def get_predictions(limit: int = 50):
    try:
        with engine.connect() as conn:
            result = conn.execute(text(f"""
                SELECT * FROM flight_price_predictions
                ORDER BY created_at DESC
                LIMIT {limit}
            """))
            rows = result.fetchall()
            columns = result.keys()
            data = [dict(zip(columns, row)) for row in rows]
        return {"predictions": data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))