
# IMPORT LIBRARIES


import os
import numpy as np
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

import warnings
warnings.filterwarnings("ignore")



BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(BASE_DIR, "flights.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODEL_DIR, "flight_price_main.joblib")
ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoders.joblib")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.joblib")


# LOAD DATA


print("Loading dataset...")
flights_df = pd.read_csv(DATA_PATH)


# FEATURE ENGINEERING


flights_df['date'] = pd.to_datetime(flights_df['date'])
flights_df['month'] = flights_df['date'].dt.month
flights_df['year'] = flights_df['date'].dt.year


# ENCODING


categorical_cols = ['from', 'to', 'flightType', 'agency']

label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    flights_df[col] = le.fit_transform(flights_df[col])
    label_encoders[col] = le


# FEATURE SELECTION


FEATURE_COLUMNS = [
    'from',
    'to',
    'flightType',
    'agency',
    'month',
    'year'
]

X = flights_df[FEATURE_COLUMNS]
y = flights_df['price']

# SCALING


scaler = StandardScaler()
X = scaler.fit_transform(X)


# TRAIN TEST SPLIT


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# MODEL TRAINING


rf_model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

rf_model.fit(X_train, y_train)


# EVALUATION


preds = rf_model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, preds))

print("RMSE:", rmse)


# MLFLOW TRACKING


mlflow.set_experiment("Flight Price Prediction")

with mlflow.start_run():
    mlflow.log_param("model", "RandomForest")
    mlflow.log_metric("RMSE", rmse)
    mlflow.sklearn.log_model(rf_model, "model")

print("MLflow tracking completed")

# SAVE ARTIFACTS


joblib.dump(rf_model, MODEL_PATH)
joblib.dump(label_encoders, ENCODER_PATH)
joblib.dump(scaler, SCALER_PATH)

print("Model saved at:", MODEL_PATH)
print("Encoders saved at:", ENCODER_PATH)
print("Scaler saved at:", SCALER_PATH)

print("Training Completed Successfully!")
