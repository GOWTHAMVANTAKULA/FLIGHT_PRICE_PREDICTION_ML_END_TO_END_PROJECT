import streamlit as st
import requests
import pandas as pd
from datetime import datetime
import os
from dotenv import load_dotenv

# -------------------------------
# Load environment variables from .env file
# -------------------------------
load_dotenv()

# -------------------------------
# FastAPI URL – read from .env
# -------------------------------
API_URL = os.getenv("API_URL", "http://localhost:8000")  # fallback for local dev

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="✈️ Flight Price Prediction", layout="centered")
st.title("✈️ Flight Price Prediction App")
st.write("Enter flight details to predict the ticket price.")

# -------------------------------
# User Inputs
# -------------------------------
cities = [
    "Brasilia (DF)", "Campo Grande (MS)", "Florianopolis (SC)",
    "Natal (RN)", "Recife (PE)", "Rio de Janeiro (RJ)",
    "Salvador (BH)", "Sao Paulo (SP)"
]

col1, col2 = st.columns(2)
with col1:
    from_city = st.selectbox("From City", cities)
with col2:
    to_city = st.selectbox("To City", cities)

col3, col4 = st.columns(2)
with col3:
    flight_type = st.selectbox("Flight Type", ["economic", "firstClass", "premium"])
with col4:
    agency = st.selectbox("Booking Agency", ["CloudFy", "FlyingDrops", "Rainbow"])

col5, col6 = st.columns(2)
with col5:
    time = st.number_input("Flight Time (hours)", min_value=1.0, value=2.0)
with col6:
    distance = st.number_input("Distance (km)", min_value=1.0, value=500.0)

date = st.date_input("Flight Date")

# -------------------------------
# Validation
# -------------------------------
if from_city == to_city:
    st.warning("⚠️ From City and To City cannot be the same.")

# -------------------------------
# Predict Button
# -------------------------------
if st.button("🔮 Predict Price", use_container_width=True):

    if from_city == to_city:
        st.error("Please choose different cities.")
        st.stop()

    payload = {
        "from_city": from_city,
        "to_city": to_city,
        "flight_type": flight_type,
        "agency": agency,
        "time": time,
        "distance": distance,
        "date": str(date)
    }

    with st.spinner("Predicting flight price..."):
        try:
            response = requests.post(f"{API_URL}/predict", json=payload)
            if response.status_code == 200:
                result = response.json()
                st.success(f"💰 Predicted Flight Price: ${result['predicted_price']:,.2f}")
                st.balloons()
            else:
                st.error(f"❌ Error: {response.json().get('detail', 'Unknown error')}")
        except Exception as e:
            st.error(f"❌ Could not connect to API: {e}")

# -------------------------------
# Show Stored Predictions
# -------------------------------
st.divider()
st.subheader("📊 View Stored Predictions")

if st.button("Show Stored Data", use_container_width=True):
    try:
        response = requests.get(f"{API_URL}/predictions?limit=50")
        if response.status_code == 200:
            data = response.json().get("predictions", [])
            if data:
                df = pd.DataFrame(data)
                st.dataframe(df)
                st.download_button(
                    label="⬇️ Download as CSV",
                    data=df.to_csv(index=False),
                    file_name="flight_price_predictions.csv",
                    mime="text/csv"
                )
            else:
                st.info("No predictions stored yet.")
        else:
            st.error("Failed to fetch data from API.")
    except Exception as e:
        st.error(f"❌ Could not connect to API: {e}")


















# import streamlit as st
# import pandas as pd
# import joblib
# import numpy as np
# from sqlalchemy import create_engine
# from dotenv import load_dotenv
# import os
# from datetime import datetime

# # -------------------------------
# # Load environment variables
# # -------------------------------
# load_dotenv()
# DATABASE_URL = os.getenv("DATABASE_URL")

# if not DATABASE_URL:
#     st.error("❌ DATABASE_URL not found. Check your .env file.")
#     st.stop()

# # -------------------------------
# # PostgreSQL Connection (Render requires SSL)
# # -------------------------------
# engine = create_engine(DATABASE_URL + "?sslmode=require")

# # -------------------------------
# # Load Model Artifacts
# # -------------------------------
# model = joblib.load("xgb_flight_price_model.joblib")
# scaler = joblib.load("standard_scaler.joblib")
# feature_columns = joblib.load("feature_columns.joblib")

# # -------------------------------
# # Streamlit UI
# # -------------------------------
# st.title("✈️ Flight Price Prediction App")
# st.write("Enter flight details to predict the ticket price.")

# # -------------------------------
# # User Inputs
# # -------------------------------
# from_city = st.selectbox("From City", [
#     "Brasilia (DF)", "Campo Grande (MS)", "Florianopolis (SC)",
#     "Natal (RN)", "Recife (PE)", "Rio de Janeiro (RJ)",
#     "Salvador (BH)", "Sao Paulo (SP)"
# ])

# to_city = st.selectbox("To City", [
#     "Brasilia (DF)", "Campo Grande (MS)", "Florianopolis (SC)",
#     "Natal (RN)", "Recife (PE)", "Rio de Janeiro (RJ)",
#     "Salvador (BH)", "Sao Paulo (SP)"
# ])

# flight_type = st.selectbox("Flight Type", ["economic", "firstClass", "premium"])
# agency = st.selectbox("Booking Agency", ["CloudFy", "FlyingDrops", "Rainbow"])
# time = st.number_input("Flight Time (hours)", min_value=1.0)
# distance = st.number_input("Distance (km)", min_value=1.0)
# date = st.date_input("Flight Date")

# # -------------------------------
# # Validation
# # -------------------------------
# if from_city == to_city:
#     st.warning("⚠️ From City and To City cannot be the same.")

# # -------------------------------
# # Prediction Button
# # -------------------------------
# if st.button("Predict Price"):

#     if from_city == to_city:
#         st.error("Please choose different cities.")
#         st.stop()

#     # Prepare Data
#     data = pd.DataFrame({
#         "from": [from_city],
#         "to": [to_city],
#         "flightType": [flight_type],
#         "agency": [agency],
#         "time": [time],
#         "distance": [distance],
#         "date": [date]
#     })

#     # Feature Engineering
#     data["date"] = pd.to_datetime(data["date"])
#     data["day"] = data["date"].dt.dayofweek
#     data["month"] = data["date"].dt.month
#     data["is_weekend"] = data["day"].apply(lambda x: 1 if x >= 5 else 0)
#     data["speed"] = data["distance"] / data["time"]
#     data.drop(columns=["date"], inplace=True)

#     # Encoding
#     data = pd.get_dummies(data)
#     data = data.reindex(columns=feature_columns, fill_value=0)

#     # Scaling
#     num_cols = ['time', 'distance', 'day', 'month', 'is_weekend', 'speed']
#     data[num_cols] = scaler.transform(data[num_cols])

#     # Prediction
#     with st.spinner("Predicting flight price..."):
#         prediction = model.predict(data)[0]

#     # Prepare DB Data
#     db_data = pd.DataFrame({
#         "from_city": [from_city],
#         "to_city": [to_city],
#         "flight_type": [flight_type],
#         "agency": [agency],
#         "time": [time],
#         "distance": [distance],
#         "date": [date],
#         "predicted_price": [prediction],
#         "created_at": [datetime.now()]
#     })

#     # Save to PostgreSQL
#     try:
#         with engine.begin() as conn:
#             db_data.to_sql(
#                 "flight_price_predictions",
#                 conn,
#                 schema="public",
#                 if_exists="append",
#                 index=False
#             )

#         st.success(f"💰 Predicted Flight Price: ${prediction:,.2f}")
#         st.info("✅ Data saved to PostgreSQL successfully!")

#     except Exception as e:
#         st.warning("⚠️ Prediction done, but database is not available.")

# # -------------------------------
# # Show Stored Data
# # -------------------------------
# st.subheader("📊 View Stored Predictions")

# if st.button("Show Stored Data"):
#     try:
#         query = """
#         SELECT * 
#         FROM flight_price_predictions
#         ORDER BY created_at DESC
#         LIMIT 50
#         """
#         df = pd.read_sql(query, engine)

#         if df.empty:
#             st.info("No data found.")
#         else:
#             st.dataframe(df)

#             # Download button
#             st.download_button(
#                 label="⬇️ Download Data as CSV",
#                 data=df.to_csv(index=False),
#                 file_name="flight_price_predictions.csv",
#                 mime="text/csv"
#             )

#     except Exception as e:
#         st.error(f"❌ Unable to fetch data: {e}")