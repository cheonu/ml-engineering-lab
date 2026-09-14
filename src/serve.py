import mlflow
from contextlib import asynccontextmanager
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd

model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    print ("Loading model...")
    
    global model
    try:
        model_uri = "models:/churn-model@champion"
        model = mlflow.sklearn.load_model(model_uri, dst_path=None)
        print (f"Successfully loaded model from:{model_uri}")
    except Exception as e: 
        print (f"Error Loading Model: {e}")
    yield

app = FastAPI(title = "Churn Prediction Service" , description = "FastAPI wrapping an MLflow model", lifespan=lifespan)

class ModelInput(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float

@app.post("/predict")
async def predict(data: ModelInput):
    df = pd.DataFrame([data.model_dump()])
    label = int(model.predict(df)[0])
    probability = float(model.predict_proba(df)[0][1])
    return {"churn": label, "churn_probability": round(probability, 4)}

@app.get("/health")
async def health():
    return {"model_loaded": model is not None}


