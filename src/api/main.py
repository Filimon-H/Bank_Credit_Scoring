# src/api/main.py

from fastapi import FastAPI
from pydantic import ValidationError
from src.api.pydantic_models import CustomerFeatures, PredictionResponse
import mlflow.pyfunc

app = FastAPI(title="Credit Risk Scoring API")

# Load model from MLflow Model Registry
model_name = "FraudDetectionModel"
model_stage = "Staging"  # or "Production"
model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{model_stage}")

@app.get("/")
def home():
    return {"message": "Credit Risk Scoring API is running."}

@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerFeatures):
    input_df = customer.to_df()
    probability = model.predict(input_df)[0]
    return PredictionResponse(probability=float(probability))
