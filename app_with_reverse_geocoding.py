
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import requests
import google.generativeai as genai

# Configure Gemini API
genai.configure(api_key="AIzaSyDerEllkn5Wu6m2ljE-XR6xRSeuv83NF7M")

# Load the ML model
ml_model = joblib.load('house_price_model.pkl')

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

@app.route('/')
def home():
    return "✅ Flask is running. Use POST /predict or /summarize-location."

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    try:
        df = pd.DataFrame([{
            "Area": data["Area"],
            "Area (sq.ft.)": data["Area_sqft"],
            "BHK": data["BHK"],
            "Bathrooms": data["Bathrooms"],
            "Furnishing Status": data["Furnishing"],
            "Age of Property (years)": data["Age"],
            "Distance to School (km)": data["Distance_School"],
            "Distance to Hospital (km)": data["Distance_Hospital"],
            "Distance to Metro (km)": data["Distance_Metro"]
        }])
        prediction = ml_model.predict(df)[0]
        return jsonify({'predictedPrice': round(prediction, 2)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

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
        gemini_model = genai.GenerativeModel('models/gemini-1.5-flash')
        response = gemini_model.generate_content(prompt)
        return jsonify({'summary': response.text})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
