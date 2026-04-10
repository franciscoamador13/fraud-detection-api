# main.py
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from enum import Enum
import os
from pathlib import Path

# Define the path to the saved model file
BASE_DIR = Path(__file__).parent
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

# Define valid transaction types
class TransactionType(str, Enum):
    PAYMENT = "PAYMENT"
    TRANSFER = "TRANSFER"
    CASH_OUT = "CASH_OUT"
    DEPOSIT = "DEPOSIT"
    #Should we use others?

class InputFeatures(BaseModel):
    """Defines the structure for input data"""
    amount: float
    oldbalanceOrg: float
    newbalanceOrig: float
    oldbalanceDest: float
    newbalanceDest: float
    type: TransactionType 
    balanceDiffOriginal: float
    balanceDiffDest: float

