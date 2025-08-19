from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

dataset_path = 'latestnewdataset_adjusted.csv' if os.path.exists('latestnewdataset_adjusted.csv') else 'latestnewdataset.csv'
print(f"Training from dataset: {dataset_path}")
df = pd.read_csv(dataset_path)

target = 'Price (Lakhs)'
features = ['Area', 'Area (sq.ft.)', 'BHK', 'Bathrooms', 'Furnishing Status',
            'Distance to School (km)', 'Distance to Hospital (km)',
            'Distance to Metro (km)', 'Age of Property (years)']

X = df[features]
y = df[target]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

categorical = ['Area', 'Furnishing Status']
numerical = ['Area (sq.ft.)', 'BHK', 'Bathrooms', 'Distance to School (km)',
             'Distance to Hospital (km)', 'Distance to Metro (km)', 'Age of Property (years)']

preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical),
    ('num', StandardScaler(), numerical)
])


pipeline = Pipeline([
    ('preprocess', preprocessor),
    ('model', RandomForestRegressor(n_estimators=200, random_state=42))
])


pipeline.fit(X_train, y_train)


y_pred = pipeline.predict(X_test)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))


joblib.dump(pipeline, 'house_price_model.pkl')
print(rmse)
print(r2)

plt.figure(figsize=(8, 6))
plt.scatter(x=y_test, y=y_pred)
plt.xlabel("Actual Price (Lakhs)")
plt.ylabel("Predicted Price (Lakhs)")
plt.title("Actual vs Predicted House Prices")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  # Ideal prediction line
plt.grid(True)
plt.tight_layout()
plt.show()
