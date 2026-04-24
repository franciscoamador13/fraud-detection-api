# main.py
from fastapi.responses import RedirectResponse
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, APIRouter
from pydantic import BaseModel
from enum import Enum
import os
from pathlib import Path

# Define the path to the saved model file
BASE_DIR = Path(__file__).parent.parent  # Go up to project root    
MODEL_DIR = BASE_DIR / "model"
MODEL_PATH = MODEL_DIR / "xgboost_fraud_model.pkl"

# Load the model during application startup
try:
    model = joblib.load(MODEL_PATH)
    print(f"Model loaded successfully from {MODEL_PATH}")
except FileNotFoundError:
    print(f"Error: Model file not found at {MODEL_PATH}")
    print(f"Make sure the model exists at: {MODEL_PATH}")
    model = None
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Initialize the FastAPI app
app = FastAPI(title="Fraud Detection API")
routerv1 = APIRouter(prefix="/v1")

@app.get("/")
async def redirect_root():
    return RedirectResponse(url="/v1/")

@routerv1.get("/")
async def root():
    return {
        "message": "This is a fraud detection model based on transaction data.",
        "version": "1.0.0",
        "endpoints": {
            "docs": "/docs"
        }
    }

# Define valid transaction types
class TransactionType(str, Enum):
    PAYMENT = "PAYMENT"
    TRANSFER = "TRANSFER"
    CASH_OUT = "CASH_OUT"
    CASH_IN = "CASH_IN"
    DEBIT = "DEBIT" #NOTE: DEBIT has only 41k records in the dataset the model was trained on, so it may not perform well for this type

class InputFeatures(BaseModel): #Base model validates automatically the input data and ensures it matches the expected structure and types
    #Defines the structure for input data
    amount: float
    oldbalanceOrg: float #Sender
    newbalanceOrig: float #Sender
    oldbalanceDest: float #Receiver
    newbalanceDest: float #Receiver
    type: TransactionType 

@routerv1.post("/predict/")
async def predict(features: InputFeatures):
    if model is None:
        return {"error": "Model not loaded. Prediction cannot be performed."}
    
    balanceDiffOriginal = features.oldbalanceOrg - features.newbalanceOrig
    balanceDiffDest = features.newbalanceDest - features.oldbalanceDest

    if features.amount <= 0:
        raise HTTPException(status_code=422, detail="Amount cannot be negative.")
    
    # Prepare the input data for prediction
    input_data = pd.DataFrame([{
        "amount": features.amount,
        "oldbalanceOrg": features.oldbalanceOrg,
        "newbalanceOrig": features.newbalanceOrig,
        "oldbalanceDest": features.oldbalanceDest,
        "newbalanceDest": features.newbalanceDest,
        "type": features.type.value,  # Convert Enum to string
        "balanceDiffOriginal": balanceDiffOriginal,
        "balanceDiffDest": balanceDiffDest
    }])
    
    # Perform prediction using the loaded model
    try:
        prediction = model.predict(input_data)[0]
        return {"prediction": int(prediction)}
    except Exception as e:
        return {"error": f"Prediction failed: {e}"}

app.include_router(routerv1)