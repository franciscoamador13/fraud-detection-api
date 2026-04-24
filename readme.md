# 🚨 Fraud Detection API

A production-ready FastAPI service for detecting fraudulent financial transactions using XGBoost machine learning model.

## 🎯 Overview

This API leverages machine learning to identify potentially fraudulent transactions in real-time. Built with **FastAPI** and **XGBoost**, it analyzes transaction patterns and returns fraud probability predictions.

**Key Features:**
- Fast, scalable REST API
- Real-time fraud detection
- Automatic input validation
- Interactive Swagger documentation
- Comprehensive error handling

## ✅ Prerequisites

- Python 3.8+
- pip (Python package manager)
- Git

## 🚀 Installation

### 1. Clone the Repository
```bash
git clone https://github.com/franciscoamador13/fraud-detection-api.git
cd fraud-detection-api
```

### 2. Download the Dataset

**⚠️ IMPORTANT:** The dataset is required to train the model.

1. Download from [Kaggle - Fraud Detection Dataset](https://www.kaggle.com/datasets/amanalisiddiqui/fraud-detection-dataset?resource=download)
2. Place the file in the `data/` folder:
   ```
   data/AIML Dataset.csv
   ```

### 3. Create Virtual Environment

**Linux/macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

## 📁 Project Structure

```
fraud-detection-api/
├── README.md                    # Project documentation
├── requirements.txt             # Python dependencies
├── data/
│   └── AIML Dataset.csv        # Dataset (⬇️ download required)
├── model/
│   └── xgboost_fraud_model.pkl # Trained XGBoost model
├── notebooks/
│   └── eda.ipynb               # Exploratory Data Analysis
├── app/
│   └── main.py                 # FastAPI application
└── .gitignore
```

## 🔌 API Documentation

### Base URL
```
http://127.0.0.1:8000/v1
```

### Endpoints

#### 1. **GET `/v1/`** — API Information
Returns general information about the API.

**Response:**
```json
{
  "message": "This is a fraud detection model based on transaction data.",
  "version": "1.0.0",
  "endpoints": {
    "docs": "/docs"
  }
}
```

#### 2. **POST `/v1/predict`** — Predict Fraud
Analyzes a transaction and returns fraud prediction.

**Request Body:**
```json
{
  "amount": 1000,
  "oldbalanceOrg": 1000000000000000,
  "newbalanceOrig": 0,
  "oldbalanceDest": 1000,
  "newbalanceDest": 1000,
  "type": "CASH_OUT"
}
```

**Response:**
```json
{
  "prediction": 1
}
```

- `0` = Legitimate transaction
- `1` = Fraudulent transaction

**Transaction Types:**
- `PAYMENT`
- `TRANSFER`
- `CASH_OUT`
- `CASH_IN`
- `DEBIT` (⚠️ Low confidence — few training examples)

## 💻 Usage

### 1. Start the API Server

```bash
cd app
uvicorn main:app --reload
```

Server runs on: `http://127.0.0.1:8000`

### 2. Access Interactive Documentation

Open in your browser:
- **Swagger UI:** http://127.0.0.1:8000/docs

## 📊 Dataset

- **Source:** [Kaggle - Fraud Detection Dataset](https://www.kaggle.com/datasets/amanalisiddiqui/fraud-detection-dataset)

## 📈 Model Performance

**XGBoost Classifier**

| Metric | Value |
|--------|-------|
| Precision (Class 1) | 0.02 |
| Recall (Class 1) | 0.94 |
| F1-Score (Class 1) | 0.04 |

## 📝 Workflow

1. **Data Exploration** → `notebooks/eda.ipynb`
2. **Model Training** → Train XGBoost on dataset
3. **Model Serialization** → Save to `model/xgboost_fraud_model.pkl`
4. **API Deployment** → FastAPI loads model at startup
5. **Real-time Predictions** → POST requests return fraud predictions

## 🛠️ Technologies Used

- **FastAPI** - Web framework
- **XGBoost** - Machine learning model
- **Pandas** - Data manipulation
- **Pydantic** - Data validation
- **Joblib** - Model serialization