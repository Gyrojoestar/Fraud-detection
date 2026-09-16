from typing import List
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(title="Fraud Detection API", version="1.0")

# Load model globally
try:
    model = joblib.load("model/model.pkl")
except Exception:
    model = None
    print("no model to load.")

class TransactionRequest(BaseModel):
    # Expecting 29 features: V1-V28 + Amount
    features: List[float] = Field(..., min_length=29, max_length=29)

@app.get("/health")
def health():
    # TODO: Return status dict indicating if model is loaded
    if model is not None:
        return {"status": "healthy", "model": "loaded"}
    return {"status": "unhealthy", "model": "missing"}

@app.post("/predict")
def predict(request: TransactionRequest):
    # TODO: Check model loaded, reshape array (1, 29), calculate proba, return JSON
    if model is None:
        raise HTTPException(status_code=500, detail="Model artifact not found")
    data = np.array(request.features).reshape(1, -1)
    probability = model.predict_proba(data)[0][1]
    probability = float(probability)
    fraud_proba = round(probability, 4)
    is_fraud = bool(probability > 0.5)
    return {"Fraud probability": fraud_proba, "Is_fraud": is_fraud}
