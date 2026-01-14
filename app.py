# app.py

import pickle
from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from datetime import datetime

app = Flask(__name__)

# --- 1. Load the trained model and scaler ---
try:
    with open('xgboost_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    print("Model and scaler loaded successfully!")
except FileNotFoundError:
    print("Error: 'xgboost_model.pkl' or 'scaler.pkl' not found. Make sure they are in the same directory.")
    exit()
except Exception as e:
    print(f"Error loading model or scaler: {e}")
    exit()

# --- 2. Define the exact columns the model expects ---
# This is crucial for consistent input for one-hot encoding
# Based on your data_encoded.head(2) and preprocessing steps:
# Numerical columns that were scaled:
NUM_COLS_SCALED = ['time', 'distance', 'day', 'month', 'is_weekend', 'speed']

# Categorical columns used for one-hot encoding (drop_first=True)
# These lists must match the columns generated during training.
# Ensure the order is consistent with the `data_encoded.columns` after `drop_first=True`.

# The 'from' unique values were: 'Recife (PE)', 'Florianopolis (SC)', 'Brasilia (DF)', 'Aracaju (SE)', 'Salvador (BH)', 'Campo Grande (MS)', 'Sao Paulo (SP)', 'Natal (RN)', 'Rio de Janeiro (RJ)'
# 'Aracaju (SE)' was the alphabetically first, so it was dropped.
FROM_COLS = [
    'from_Brasilia (DF)', 'from_Campo Grande (MS)', 'from_Florianopolis (SC)',
    'from_Natal (RN)', 'from_Recife (PE)', 'from_Rio de Janeiro (RJ)',
    'from_Salvador (BH)', 'from_Sao Paulo (SP)'
]

# The 'to' unique values were: 'Florianopolis (SC)', 'Recife (PE)', 'Brasilia (DF)', 'Salvador (BH)', 'Aracaju (SE)', 'Campo Grande (MS)', 'Sao Paulo (SP)', 'Natal (RN)', 'Rio de Janeiro (RJ)'
# 'Aracaju (SE)' was dropped.
TO_COLS = [
    'to_Brasilia (DF)', 'to_Campo Grande (MS)', 'to_Florianopolis (SC)',
    'to_Natal (RN)', 'to_Recife (PE)', 'to_Rio de Janeiro (RJ)',
    'to_Salvador (BH)', 'to_Sao Paulo (SP)'
]

# 'flightType' unique values: 'firstClass', 'economic', 'premium'
# 'economic' was dropped.
FLIGHT_TYPE_COLS = ['flightType_firstClass', 'flightType_premium']

# 'agency' unique values: 'FlyingDrops', 'CloudFy', 'Rainbow'
# 'CloudFy' was dropped.
AGENCY_COLS = ['agency_FlyingDrops', 'agency_Rainbow']

# Combine all expected column names in the order the model expects
EXPECTED_MODEL_COLUMNS = NUM_COLS_SCALED + FROM_COLS + TO_COLS + FLIGHT_TYPE_COLS + AGENCY_COLS

# --- 3. Preprocessing function for new data ---
def preprocess_input(data: dict) -> pd.DataFrame:
    """
    Applies the same preprocessing steps to new input data as done during training.
    """
    # Create a DataFrame from the single input record
    df_input = pd.DataFrame([data])

    # --- Feature Engineering ---
    # Convert date string to datetime object
    df_input['date'] = pd.to_datetime(df_input['date'], errors='coerce', format='%m/%d/%Y')

    # Extract day, month, is_weekend
    df_input['day_name'] = df_input['date'].dt.day_name()
    df_input['day'] = df_input['date'].dt.dayofweek
    df_input['month_name'] = df_input['date'].dt.month_name()
    df_input['month'] = df_input['date'].dt.month
    # Note: 'is_weekend' logic as per notebook: 1 if day_of_week is Tuesday-Sunday, 0 if Monday.
    # If standard Sat/Sun weekend is desired, change to `lambda x: 1 if x >= 5 else 0`
    df_input['is_weekend'] = df_input['day'].apply(lambda x : 1 if x >= 1 else 0) 

    # Calculate speed
    # Handle division by zero for time, if time can be 0. Default to 0 or a very small number.
    df_input['speed'] = df_input['distance'] / df_input['time'].replace(0, np.nan) # Replace 0 with NaN to avoid /0
    df_input['speed'] = df_input['speed'].fillna(0) # Fill NaN from /0 with 0, or some other sensible default

    print("\n--- Raw input with engineered features ---")
    print(df_input)

    # --- Robust One-Hot Encoding and Column Alignment ---
    # Create an empty DataFrame with all EXPECTED_MODEL_COLUMNS, initialized to 0
    # This ensures consistency even if a category is missing in the current input
    processed_df = pd.DataFrame(0, index=[0], columns=EXPECTED_MODEL_COLUMNS)

    # Populate numerical columns directly
    for col in NUM_COLS_SCALED:
        if col in df_input.columns:
            processed_df[col] = df_input[col].iloc[0]

    # Manually set one-hot encoded columns based on input values
    # 'from' column
    from_val = data['from']
    if from_val != 'Aracaju (SE)': # 'Aracaju (SE)' is the dropped reference category
        col_name = f"from_{from_val}"
        if col_name in processed_df.columns:
            processed_df[col_name] = 1

    # 'to' column
    to_val = data['to']
    if to_val != 'Aracaju (SE)': # 'Aracaju (SE)' is the dropped reference category
        col_name = f"to_{to_val}"
        if col_name in processed_df.columns:
            processed_df[col_name] = 1

    # 'flightType' column
    flight_type_val = data['flightType']
    if flight_type_val != 'economic': # 'economic' is the dropped reference category
        col_name = f"flightType_{flight_type_val}"
        if col_name in processed_df.columns:
            processed_df[col_name] = 1

    # 'agency' column
    agency_val = data['agency']
    if agency_val != 'CloudFy': # 'CloudFy' is the dropped reference category
        col_name = f"agency_{agency_val}"
        if col_name in processed_df.columns:
            processed_df[col_name] = 1

    print("\n--- After manual one-hot encoding and column alignment (before scaling) ---")
    print(processed_df)

    # --- Data Scaling ---
    # Scale numerical features using the loaded scaler
    # Ensure only NUM_COLS_SCALED are passed to the scaler
    processed_df[NUM_COLS_SCALED] = scaler.transform(processed_df[NUM_COLS_SCALED])

    print("\n--- Final processed data for prediction (after scaling) ---")
    print(processed_df)

    return processed_df

# --- 4. Prediction endpoint ---
@app.route('/predict', methods=['POST'])
def predict():
    if not request.json:
        return jsonify({"error": "Please send JSON data"}), 400

    try:
        data = request.json
        # Preprocess the input data
        processed_data = preprocess_input(data)

        # Make prediction
        prediction = model.predict(processed_data)

        # Return the prediction as JSON
        return jsonify({"predicted_price": prediction[0].item()}) # .item() converts numpy float to Python float

    except KeyError as e:
        return jsonify({"error": f"Missing input feature: {e}. Please ensure all required features are provided."}), 400
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {e}"}), 500

# --- 5. Run the Flask app ---
if __name__ == '__main__':
    # Set use_reloader=False if you encounter issues with multiple restarts in some environments
    app.run(debug=True, use_reloader=True)
