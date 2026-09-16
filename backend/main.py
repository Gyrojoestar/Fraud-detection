from typing import List
import os
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import mlflow

app = FastAPI(title="Fraud Detection API", version="1.0")

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///D:/fraud-detection/mlflow.db"))
exp = mlflow.get_experiment_by_name("fraud_detection_xgboost")
if exp is None:
    raise RuntimeError("MLflow experiment not found. Run train.py first.")

sorted_runs = mlflow.search_runs(
    experiment_ids=[exp.experiment_id], 
    order_by=["metrics.pr_auc DESC"]
)

if sorted_runs.empty:
    raise RuntimeError("No logged runs found in MLflow.")


top_model = sorted_runs.iloc[0]
run_id = top_model['run_id']
model_path = f"runs:/{run_id}/model"

model = mlflow.xgboost.load_model(model_path)

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
