import random
from typing import List
import json
import os
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, Field
import mlflow
from contextlib import asynccontextmanager
import backend.database.conn as dblib
from sqlalchemy.orm import Session

model = None

# default stuff
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model # initialised to None
    
    try:
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
        print(mlflow.get_tracking_uri())
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
        top_run_id = top_model['run_id']
        # "model" is stored in the metadata not an actual subfolder
        model = mlflow.xgboost.load_model(f"runs:/{top_run_id}/model")

    except Exception as e:
        print(f"Warning: failed to load model. {e}")
        #dont put checks here 
        
    
    # everything below yield is ran after the server is shut down
    yield
    model = None # clear up memory
    
    
app = FastAPI(title="Fraud Detection API", version="1.0", lifespan=lifespan)

def db_get():
    db = dblib.SessionLocal()
    try:
        yield db
    finally:
        db.close()  

class TransactionRequest(BaseModel):
    # Expecting 29 features: V1-V28 + Amount
    features: List[float] = Field(..., min_length=29, max_length=29)
    
    @property
    def amount(self) -> float:
        return self.features[-1]  # The 29th element

    @property
    def pca_features(self) -> List[float]:
        return self.features[:28] # V1 through V28

@app.get("/health")
def health():
    # TODO: Return status dict indicating if model is loaded
    if model is not None:
        return {"status": "healthy", "model": "loaded"}
    return {"status": "unhealthy", "model": "missing"}

@app.post("/predict")
def predict(request: TransactionRequest, db: Session = Depends(db_get)):
    # TODO: Check model loaded, reshape array (1, 29), calculate proba, return JSON
    if model is None:
        raise HTTPException(status_code=500, detail="Model artifact not found")
    data = np.array(request.features).reshape(1, -1)
    probability = model.predict_proba(data)[0][1]
    probability = float(probability)
    fraud_proba = round(probability, 4)
    is_fraud = bool(probability > 0.5)
    try:
        new_transaction = dblib.RawTransaction(
            amount=request.amount,
            card_class=True, # Or pull this from a separate parameter if needed
            **{f"v{i+1}": float(val) for i, val in enumerate(request.pca_features)}
        )

        db.add(new_transaction)
        
        db.flush() 
        
        new_prediction = dblib.ModelPred(
            transaction_id=new_transaction.transaction_id,
            pred_class=is_fraud, 
            confidence_score=fraud_proba
        )
        db.add(new_prediction)
        db.commit()
        
    except Exception as e:
        db.rollback() 
        raise HTTPException(status_code=500, detail=f"Database storage failed: {str(e)}")
    
    return fraud_proba, is_fraud, request.amount
    
@app.api_route("/test-add-transaction", methods=["GET", "POST"])
def test_add_transaction(db: Session = Depends(db_get)):
    """
    Reads a random row from 'creditCardRealTime.csv' (or creditcard.csv) 
    and posts it through the transaction flow to save in Supabase.
    """
    csv_path = "card_cleaned_keep.csv"

    if not os.path.exists(csv_path):
        raise HTTPException(status_code=404, detail=f"CSV file '{csv_path}' not found.")

    df = pd.read_csv(csv_path)
    random_idx = random.randint(0, len(df) - 1)
    row = df.iloc[random_idx]

    # save to supabase raw_transaction table
    # change from np datatype to python datatype for supabase storage
    try:
        payload = {
            "amount": float(row["amount"]),
            "card_class": bool(int(row["class"])),
            **{f"v{i+1}": float(row[f"v{i+1}"]) for i in range(28)}
        }

        new_transaction = dblib.RawTransaction(**payload)
        db.add(new_transaction)
        db.flush()
        db.commit()

    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Database storage failed: {str(e)}")

    return {"status": "ok", "message": "test route reached", "row_index": int(random_idx), "row_data": row.to_dict()}
    
