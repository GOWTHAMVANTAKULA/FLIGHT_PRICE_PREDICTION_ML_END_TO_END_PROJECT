import pickle
from flask import Flask, request, jsonify, render_template # Added render_template
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
NUM_COLS_SCALED = ['time', 'distance', 'day', 'month', 'is_weekend', 'speed']

FROM_COLS = [
    'from_Brasilia (DF)', 'from_Campo Grande (MS)', 'from_Florianopolis (SC)',
    'from_Natal (RN)', 'from_Recife (PE)', 'from_Rio de Janeiro (RJ)',
    'from_Salvador (BH)', 'from_Sao Paulo (SP)'
]

TO_COLS = [
    'to_Brasilia (DF)', 'to_Campo Grande (MS)', 'to_Florianopolis (SC)',
    'to_Natal (RN)', 'to_Recife (PE)', 'to_Rio de Janeiro (RJ)',
    'to_Salvador (BH)', 'to_Sao Paulo (SP)'
]

FLIGHT_TYPE_COLS = ['flightType_firstClass', 'flightType_premium']

AGENCY_COLS = ['agency_FlyingDrops', 'agency_Rainbow']

EXPECTED_MODEL_COLUMNS = NUM_COLS_SCALED + FROM_COLS + TO_COLS + FLIGHT_TYPE_COLS + AGENCY_COLS

# --- 3. Preprocessing function for new data from form ---
def preprocess_input(form_data) -> pd.DataFrame:
    """
    Applies the same preprocessing steps to new input data from an HTML form.
    """
    # Prepare input dictionary for DataFrame creation
    input_dict = {
        'from': form_data.get('from'),
        'to': form_data.get('to'),
        'flightType': form_data.get('flightType'),
        'time': float(form_data.get('time')),
        'distance': float(form_data.get('distance')),
        'agency': form_data.get('agency'),
        'date': form_data.get('date') # Assuming 'MM/DD/YYYY' format from HTML input type="date"
    }

    df_input = pd.DataFrame([input_dict])

    # --- Feature Engineering ---
    # HTML date input type="date" gives YYYY-MM-DD format
    df_input['date'] = pd.to_datetime(df_input['date'], errors='coerce', format='%Y-%m-%d') 

    df_input['day'] = df_input['date'].dt.dayofweek
    df_input['month'] = df_input['date'].dt.month
    # Note: 'is_weekend' logic as per notebook: 1 if day_of_week is Tuesday-Sunday, 0 if Monday.
    df_input['is_weekend'] = df_input['day'].apply(lambda x : 1 if x >= 1 else 0) 

    # Calculate speed
    # Handle division by zero for time, if time can be 0. Default to 0.
    df_input['speed'] = df_input['distance'] / df_input['time'].replace(0, np.nan) 
    df_input['speed'] = df_input['speed'].fillna(0) 

    # --- Robust One-Hot Encoding and Column Alignment ---
    # Create an empty DataFrame with all EXPECTED_MODEL_COLUMNS, initialized to 0
    processed_df = pd.DataFrame(0, index=[0], columns=EXPECTED_MODEL_COLUMNS)

    # Populate numerical columns directly
    for col in NUM_COLS_SCALED:
        if col in df_input.columns:
            processed_df[col] = df_input[col].iloc[0]

    # Manually set one-hot encoded columns based on input values
    from_val = input_dict['from']
    if from_val != 'Aracaju (SE)': # 'Aracaju (SE)' was the alphabetically first, thus dropped during training
        col_name = f"from_{from_val}"
        if col_name in processed_df.columns:
            processed_df[col_name] = 1

    to_val = input_dict['to']
    if to_val != 'Aracaju (SE)': # 'Aracaju (SE)' was the alphabetically first of the 'to' cities, thus dropped
        col_name = f"to_{to_val}"
        if col_name in processed_df.columns:
            processed_df[col_name] = 1

    flight_type_val = input_dict['flightType']
    if flight_type_val != 'economic': # 'economic' was dropped as it's alphabetically first
        col_name = f"flightType_{flight_type_val}"
        if col_name in processed_df.columns:
            processed_df[col_name] = 1

    agency_val = input_dict['agency']
    if agency_val != 'CloudFy': # 'CloudFy' was dropped as it's alphabetically first
        col_name = f"agency_{agency_val}"
        if col_name in processed_df.columns:
            processed_df[col_name] = 1

    # --- Data Scaling ---
    # Scale numerical features using the loaded scaler
    # Ensure only NUM_COLS_SCALED are passed to the scaler
    processed_df[NUM_COLS_SCALED] = scaler.transform(processed_df[NUM_COLS_SCALED])

    return processed_df

# --- 4. Define routes ---
@app.route('/')
def home():
    # Render the HTML template for the home page, passing empty/None values initially
    return render_template('index.html', form_data={}, predicted_price=None, error_message=None)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data directly from request.form
        form_data = request.form.to_dict() # Convert ImmutableMultiDict to a regular dict

        # Preprocess the input data
        processed_data = preprocess_input(form_data)

        # Make prediction
        prediction = model.predict(processed_data)[0]

        # Format prediction for display
        predicted_price = f"{prediction:.2f}"

        # Render the HTML template with the prediction and original form data
        # Pass form_data back to pre-fill the form fields
        return render_template('index.html', predicted_price=predicted_price, form_data=form_data)

    except KeyError as e:
        # Pass form_data back on error too, so user doesn't lose input
        return render_template('index.html', error_message=f"Missing input: {e}. Please ensure all required fields are filled.", form_data=request.form.to_dict())
    except ValueError as e:
        return render_template('index.html', error_message=f"Invalid data: {e}. Please check numerical entries and date format.", form_data=request.form.to_dict())
    except Exception as e:
        return render_template('index.html', error_message=f"Prediction failed: {e}", form_data=request.form.to_dict())

# --- 5. Run the Flask app ---
if __name__ == '__main__':
    # Set use_reloader=True to automatically restart the server on code changes
    app.run(debug=True, use_reloader=True)
