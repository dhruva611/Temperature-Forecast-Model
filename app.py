import os
import json
import pandas as pd
import numpy as np
import joblib
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from tensorflow.keras.models import load_model
from datetime import datetime

# --- 1. CRITICAL FIX FOR PICKLE ERROR ---
# Import the class from the file sitting next to app.py
from feature_engineer import FeatureEngineer

# 🩹 THE MAGIC FIX:
# We assign the imported class to the current script's namespace (__main__).
# This tricks pickle into finding the class where it expects it.
import __main__
__main__.FeatureEngineer = FeatureEngineer

# --- 2. App Setup ---
app = Flask(__name__)
CORS(app)

# Define paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'models')

# --- 3. Load Artifacts ---
print("Loading artifacts...")
try:
    # Load Metadata
    with open(os.path.join(MODEL_DIR, 'feature_metadata.json'), 'r') as f:
        metadata = json.load(f)

    # Load Preprocessing Pipeline
    # Because of the fix above, this line will now work!
    pipeline = joblib.load(os.path.join(MODEL_DIR, 'preprocess.pkl'))

    # Load Keras Model
    model = load_model(os.path.join(MODEL_DIR, 'model.keras'))
    
    print("✅ Artifacts loaded successfully.")
except Exception as e:
    print(f"❌ Error loading artifacts: {e}")
    # We don't exit here just so you can see the error on the webpage if needed,
    # but usually, you should exit.

# --- 4. Helper: Time Formatter ---
def format_time_12h(time_str):
    """Converts '14:30' (HTML input) to '02:30 PM' (Model format)"""
    if not time_str: return None
    try:
        d = datetime.strptime(time_str, "%H:%M")
        return d.strftime("%I:%M %p")
    except:
        return time_str

# --- 5. Routes ---

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        # 1. Extract Date Time
        date_time_str = data.get('date_time')
        # Replace 'T' with space if coming from datetime-local input
        if 'T' in date_time_str:
            date_time_str = date_time_str.replace('T', ' ')
            
        dt_obj = pd.to_datetime(date_time_str, dayfirst=True)

        # 2. Prepare Feature Dictionary
        input_data = {
            'date_time': [dt_obj],
            'DewPointC': [float(data['DewPointC'])],
            'humidity': [float(data['humidity'])],
            'cloudcover': [float(data['cloudcover'])],
            'uvIndex': [int(data['uvIndex'])],
            'sunHour': [float(data['sunHour'])],
            'precipMM': [float(data['precipMM'])],
            'pressure': [float(data['pressure'])],
            'windspeedKmph': [float(data['windspeedKmph'])],
            # Handle time conversion here
            'sunrise': [format_time_12h(data['sunrise'])],
            'sunset': [format_time_12h(data['sunset'])]
        }

        # 3. Create DataFrame
        df = pd.DataFrame(input_data)

        # 4. Apply Index Logic (Critical for your FeatureEngineer)
        df.index = pd.to_datetime(df['date_time'])
        df = df.drop(columns=['date_time'])

        # 5. Preprocess
        processed_data = pipeline.transform(df)

        # 6. Predict
        prediction = model.predict(processed_data)
        predicted_temp = float(prediction[0][0])

        return jsonify({
            "status": "success",
            "predicted_temperature": round(predicted_temp, 2),
            "input_date": str(dt_obj)
        })

    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)