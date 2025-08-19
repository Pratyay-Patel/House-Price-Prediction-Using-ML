
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS

from flask_cors import CORS
import joblib
import pandas as pd
import requests
import os
try:
    import google.generativeai as genai
except Exception:
    genai = None

# Configure Gemini API (use environment variable if available)
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GEMINI_ENABLED = bool(GEMINI_API_KEY and genai is not None)
if GEMINI_ENABLED:
    genai.configure(api_key=GEMINI_API_KEY)

# Global, dynamic prediction scale (can be changed at runtime)
CURRENT_PREDICTION_SCALE = float(os.environ.get('PREDICTION_SCALE', '1.0'))

# Load the ML model (with auto-reload on file change)
ML_MODEL_PATH = 'house_price_model.pkl'
_ml_model = None
_ml_model_mtime = None

def get_ml_model():
    global _ml_model, _ml_model_mtime
    try:
        mtime = os.path.getmtime(ML_MODEL_PATH)
    except Exception:
        mtime = None
    if _ml_model is None or (_ml_model_mtime is not None and mtime is not None and mtime != _ml_model_mtime):
        _ml_model = joblib.load(ML_MODEL_PATH)
        _ml_model_mtime = mtime
    elif _ml_model is None and mtime is not None:
        _ml_model = joblib.load(ML_MODEL_PATH)
        _ml_model_mtime = mtime
    return _ml_model

app = Flask(__name__)
CORS(app)

# Reverse geocoding using OpenStreetMap Nominatim
def reverse_geocode(lat, lng):
    try:
        url = "https://nominatim.openstreetmap.org/reverse"
        params = {
            "format": "json",
            "lat": lat,
            "lon": lng,
            "zoom": 14,
            "addressdetails": 1
        }
        headers = {"User-Agent": "RealEstateAI/1.0"}
        response = requests.get(url, params=params, headers=headers)
        if response.status_code == 200:
            return response.json().get("display_name", "Pune")
        return "Pune"
    except:
        return "Pune"

# @app.route('/')
# def home():
#     return "✅ Flask is running. Use POST /predict or /summarize-location."


@app.route("/")
def home():
    # Serve HTML as bytes to avoid Windows default cp1252 decoding issues
    return send_file("index_with_magicwand.html", mimetype="text/html; charset=utf-8")


@app.before_request
def update_scale_from_query():
    global CURRENT_PREDICTION_SCALE
    if 'scale' in request.args:
        try:
            new_scale = float(request.args.get('scale'))
            CURRENT_PREDICTION_SCALE = new_scale
        except Exception:
            pass


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json() or {}

    def pick(*keys, default=None, required=False):
        for key in keys:
            if key in data and data[key] is not None:
                return data[key]
        if required:
            raise KeyError(f"Missing required fields: one of {keys}")
        return default

    try:
        # Accept multiple key variants from the frontend and map to training feature names
        area = pick("Area", "area", required=True)
        area_sqft = pick("Area (sq.ft.)", "Area_sqft", "area_sqft", required=True)
        bhk = pick("BHK", "bhk", required=True)
        bathrooms = pick("Bathrooms", "bathrooms", required=True)
        furnishing = pick("Furnishing Status", "Furnishing", "furnishing", required=True)
        age_years = pick("Age of Property (years)", "Age", "age", required=True)
        dist_school = pick("Distance to School (km)", "Distance_School", "distance_school", required=True)
        dist_hospital = pick("Distance to Hospital (km)", "Distance_Hospital", "distance_hospital", required=True)
        dist_metro = pick("Distance to Metro (km)", "Distance_Metro", "distance_metro", required=True)

        # Coerce types where applicable
        def to_float(value):
            try:
                return float(value)
            except Exception:
                return value

        df = pd.DataFrame([{
            "Area": str(area),
            "Area (sq.ft.)": to_float(area_sqft),
            "BHK": to_float(bhk),
            "Bathrooms": to_float(bathrooms),
            "Furnishing Status": str(furnishing),
            "Age of Property (years)": to_float(age_years),
            "Distance to School (km)": to_float(dist_school),
            "Distance to Hospital (km)": to_float(dist_hospital),
            "Distance to Metro (km)": to_float(dist_metro)
        }])

        prediction = get_ml_model().predict(df)[0]
        # Optional server-side multiplier to quickly scale displayed prices without retraining
        prediction_scale = CURRENT_PREDICTION_SCALE
        raw_prediction = float(prediction)
        scaled_prediction = raw_prediction * prediction_scale
        return jsonify({
            'predictedPrice': round(scaled_prediction, 2),
            'rawPrediction': round(raw_prediction, 2),
            'scaleApplied': prediction_scale
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/summarize-location', methods=['POST'])
def summarize_location():
    data = request.get_json()
    lat = data.get('lat')
    lng = data.get('lng')

    if not lat or not lng:
        return jsonify({'error': 'Missing coordinates'}), 400

    location = reverse_geocode(lat, lng)

    prompt = f"""Give me a real estate summary in under 150 words for:
Location: {location}, Coordinates: ({lat}, {lng}), in Pune, India.
Mention:
- Nearby locality
- Average property price per square foot in INR
- Estimated rent for 1BHK and 3BHK
- Any development trends or demand insights"""

    try:
        if not GEMINI_ENABLED:
            return jsonify({'summary': f"Location: {location}. Coordinates: ({lat}, {lng}). Gemini API key not configured on server. Please set GEMINI_API_KEY to enable AI summary."})
        gemini_model = genai.GenerativeModel('models/gemini-1.5-flash')
        response = gemini_model.generate_content(prompt)
        return jsonify({'summary': response.text})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/set-scale', methods=['POST'])
def set_scale():
    global CURRENT_PREDICTION_SCALE
    data = request.get_json() or {}
    try:
        new_scale = float(data.get('scale'))
        CURRENT_PREDICTION_SCALE = new_scale
        return jsonify({'ok': True, 'predictionScale': CURRENT_PREDICTION_SCALE})
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)}), 400


@app.route('/get-config', methods=['GET'])
def get_config():
    return jsonify({'predictionScale': CURRENT_PREDICTION_SCALE})

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
