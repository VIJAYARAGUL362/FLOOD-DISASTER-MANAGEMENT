import joblib
from fastapi import FastAPI, HTTPException
import numpy as np
from pydantic import BaseModel

app = FastAPI()

# Define input data model
class PredictionInput(BaseModel):
    features: list[float]

# Load the model
with open('Random_forest_model.pkl', 'rb') as file:
    model = joblib.load(file)

@app.post("/predict")
async def predict(input_data: PredictionInput):
    try:
        features = np.array([input_data.features])
        prediction = model.predict(features)
        return {"prediction": prediction.tolist()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))