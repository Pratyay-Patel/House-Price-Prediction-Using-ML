# House-Price-Prediction-Using-ML

Predicting House Prices in Pune using Random Forest

## Quickstart (Demo-ready)

1) Create and activate a virtual environment (optional but recommended)

- Windows PowerShell
  - `python -m venv venv`
  - `./venv/Scripts/Activate.ps1`

2) Install dependencies

```bash
pip install -r requirements.txt
```

3) Train or reuse the model

- The repo already includes `house_price_model.pkl`. If you want to retrain:

```bash
python train_model.py
```

4) Run the web app

```bash
python app_with_reverse_geocoding.py
```

Then open your browser to `http://127.0.0.1:5000/`.

## Features

- Great UI with a price prediction form and interactive map
- Predicts price (in Lakhs) from inputs: `Area`, `Area (sq.ft.)`, `BHK`, `Bathrooms`, `Furnishing Status`, `Age of Property (years)`, and distances to School/Hospital/Metro
- Optional AI-powered location summaries on map clicks

## Optional: Enable AI location summaries (Gemini)

The app can summarize clicked map locations using Gemini. This is optional.

1) Get an API key
   - Create a key and set it as an environment variable `GEMINI_API_KEY`.
2) Run the app with the environment variable set

```powershell
$env:GEMINI_API_KEY="YOUR_KEY_HERE"; python app_with_reverse_geocoding.py
```

If no key is provided, the app will still run and show a simple fallback summary.

## Notes

- The frontend calls relative endpoints (`/predict` and `/summarize-location`), so it works as long as the HTML is served by Flask (default behavior in `app_with_reverse_geocoding.py`).
- CORS is enabled; you can also host the HTML separately if needed.
