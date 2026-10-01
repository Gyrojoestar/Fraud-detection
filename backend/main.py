import random
from typing import List
import json
import os
from pathlib import Path
import math
import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager
import backend.database.conn as dblib
from sqlalchemy.orm import Session
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

model = None

# default stuff
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model # initialised to None
    
    try:
        model_path = Path(__file__).resolve().parent.parent / "model.ubj"
        model = XGBClassifier()
        model.load_model(model_path)

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
    
def clean_parameter_json(params: dict) -> dict:
    """
    Recursively convert numpy data types in the parameters dictionary to native Python types
    and removes any invalid values
    """
    def convert_value(value):
        if isinstance(value, np.generic):
            return value.item()  # Convert numpy scalar to native Python type
        elif isinstance(value, dict):
            return {k: convert_value(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [convert_value(v) for v in value]
        else:
            return value

    cleaned_params = convert_value(params)
    
    # Remove any invalid values (e.g., NaN, inf)
    def remove_invalid_values(d):
        if isinstance(d, dict):
            return {k: remove_invalid_values(v) for k, v in d.items() if not (isinstance(v, float) and (math.isnan(v) or math.isinf(v)))}
        elif isinstance(d, list):
            return [remove_invalid_values(v) for v in d if not (isinstance(v, float) and (math.isnan(v) or math.isinf(v)))]
        else:
            return d

    cleaned_params = remove_invalid_values(cleaned_params)
    
    return cleaned_params
    
@app.get("/")
def root():
    ui_path = Path(__file__).resolve().parent / "static" / "index.html"
    if ui_path.exists():
        return FileResponse(ui_path)
    
    # Fallback JSON if index.html is missing
    return {
        "message": "Welcome to the Fraud Detection API.",
        "available_commands": {
            "/predict": "POST endpoint to get fraud prediction for a transaction.",
            "/health": "GET endpoint to check if the model is loaded and healthy.",
            "/model-info": "GET endpoint to retrieve model parameters and feature importance.",
            "/test-add-transaction": "GET/POST endpoint to test adding a random transaction from the CSV file.",
            "/docs": "Interactive API documentation."
        }
    }
@app.get("/health")
def health():
    # TODO: Return status dict indicating if model is loaded
    if model is not None:
        return {"status": "healthy", "model": "loaded"}
    return {"status": "unhealthy", "model": "missing"}

@app.get("/model-info")
def model_info():
    if model is None:
        raise HTTPException(status_code=500, detail="Model artifact not found")
    
    model_params = clean_parameter_json(model.get_params())
    
    # Return model parameters and feature importance
    return {
        "model_params": model_params
    }

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
            card_class=is_fraud,
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
        db.commit()

    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Database storage failed: {str(e)}")

    return {"status": "ok", "message": "test route reached", "row_index": int(random_idx), "row_data": row.to_dict()}
    
