import joblib
from fastapi import FastAPI, HTTPException
import numpy as np
from pydantic import BaseModel

app = FastAPI()

class PredictionInput(BaseModel):
    features: list[float]


# Load models
with open('Random_forest_model.pkl', 'rb') as file:
    model = joblib.load(file)

with open('XGBOOST.pkl', 'rb') as file:
    model2 = joblib.load(file)

def transform_features(features: list[float]) -> np.ndarray:
    """Convert lat/lon to sin/cos and keep rest of features."""
    lat, lon = features[0], features[1]
    lat_sin, lat_cos = np.sin(np.radians(lat)), np.cos(np.radians(lat))
    lon_sin, lon_cos = np.sin(np.radians(lon)), np.cos(np.radians(lon))

    # Build final array
    arr = [lat_sin, lat_cos, lon_sin, lon_cos] + features[2:]
    return np.array(arr).reshape(1, -1)

@app.post("/predict/happen")
async def predict_happen(input_data: PredictionInput):
    try:
        arr = transform_features(input_data.features)
        prediction = model.predict(arr)
        return {"prediction": prediction.tolist()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/predict/damage")
async def predict_damage(input_data: PredictionInput):
    try:
        arr = transform_features(input_data.features)
        prediction = model2.predict(arr)  # <- Correct model used
        return {"prediction": prediction.tolist()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.api_route("/health", methods=["GET", "HEAD"])
async def health_check():
    return {"status": "alive", "service": "flood-disaster-management"}