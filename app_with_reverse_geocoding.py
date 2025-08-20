
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS

from flask_cors import CORS
import joblib
import pandas as pd
import requests
import os
import math
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

app = Flask(__name__, static_folder=".", static_url_path="")
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
    return send_file("index_with_magicwand.html", mimetype="text/html; charset=utf-8")

@app.route("/<path:path>")
def static_proxy(path):
    return send_file(path)


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
            # Deterministic location-aware fallback summary
            def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
                R = 6371.0
                dlat = math.radians(lat2 - lat1)
                dlon = math.radians(lon2 - lon1)
                a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
                c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
                return R * c

            def direction_from_center(lat1: float, lon1: float, lat2: float, lon2: float) -> str:
                dy = lat2 - lat1
                dx = lon2 - lon1
                vert = 'north' if dy > 0 else 'south'
                hori = 'east' if dx > 0 else 'west'
                if abs(dy) < 0.005:
                    vert = ''
                if abs(dx) < 0.005:
                    hori = ''
                if vert and hori:
                    return f"{vert}-{hori}"
                return vert or hori or 'central'

            pune_lat, pune_lng = 18.5204, 73.8567
            dist_km = haversine_km(pune_lat, pune_lng, float(lat), float(lng))
            bearing = direction_from_center(pune_lat, pune_lng, float(lat), float(lng))

            loc_lower = (location or '').lower()
            # Profiles for common Pune sub-markets
            profiles = [
                (['hinjewadi', 'wakad', 'baner', 'balewadi'],
                 {'psf': '6,500–9,500', 'rent1': '₹16k–₹24k', 'rent3': '₹38k–₹60k', 'note': 'Strong IT/office demand, active rental market'}),
                (['viman nagar', 'kalyani nagar', 'vimannagar'],
                 {'psf': '7,500–11,000', 'rent1': '₹20k–₹28k', 'rent3': '₹45k–₹75k', 'note': 'Airport proximity, retail hubs, premium demand'}),
                (['kothrud', 'karve nagar', 'paud road', 'ideal colony'],
                 {'psf': '7,000–9,500', 'rent1': '₹18k–₹24k', 'rent3': '₹40k–₹60k', 'note': 'Established residential, good schools and connectivity'}),
                (['aundh', 'pashan'],
                 {'psf': '7,500–10,500', 'rent1': '₹19k–₹26k', 'rent3': '₹42k–₹65k', 'note': 'Mature residential, near employment corridors'}),
                (['hadapsar', 'magarpatta', 'amanora', 'kharadi'],
                 {'psf': '6,800–9,800', 'rent1': '₹17k–₹24k', 'rent3': '₹38k–₹58k', 'note': 'IT/SEZ driven demand; modern townships'}),
                (['kondhwa', 'nibm', 'bibwewadi', 'katraj'],
                 {'psf': '5,200–7,500', 'rent1': '₹12k–₹19k', 'rent3': '₹28k–₹45k', 'note': 'Developing pockets with improving social infra'})
            ]

            matched = None
            for keys, prof in profiles:
                if any(k in loc_lower for k in keys):
                    matched = prof
                    break

            if not matched:
                # Zone-based fallback using distance from center
                if dist_km <= 5:
                    matched = {'psf': '8,500–12,000', 'rent1': '₹20k–₹30k', 'rent3': '₹45k–₹75k', 'note': 'Central city convenience; steady demand'}
                elif dist_km <= 10:
                    matched = {'psf': '6,500–9,000', 'rent1': '₹15k–₹22k', 'rent3': '₹35k–₹55k', 'note': 'Balanced affordability and access'}
                else:
                    matched = {'psf': '4,500–7,000', 'rent1': '₹10k–₹18k', 'rent3': '₹25k–₹40k', 'note': 'Peripheral belt; value buys and ongoing development'}

            fallback = (
                f"Location: {location}. ~{dist_km:.1f} km {bearing} of Pune center. "
                f"Avg price: {matched['psf']} per sq.ft. "
                f"Typical rent: 1BHK {matched['rent1']}, 3BHK {matched['rent3']}. "
                f"Trend: {matched['note']}."
            )
            return jsonify({'summary': fallback})
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
    return jsonify({})

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
