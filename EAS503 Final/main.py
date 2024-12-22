from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

# Load the model
model = joblib.load('final_model.joblib')

class InputData(BaseModel):
    features: list

@app.post("/predict")
async def predict(data: InputData):
    features = np.array(data.features).reshape(1, -1)
    prediction = model.predict(features)
    return {"prediction": int(prediction[0])}

@app.get("/")
async def root():
    return {"message": "Income Prediction API"}
