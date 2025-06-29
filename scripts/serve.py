from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel, Field
from typing import List
import pandas as pd
import yaml
import json
from utils.preprocessing import preprocess_data
from src.model import CatBoostPipeline
from sklearn.metrics import classification_report

# ---- Initialization ----

app = FastAPI(
    title="CatBoost Model API",
    description="API for single/batch prediction with a CatBoost model. See /docs for JSON schema.",
    version="1.0"
)

MODEL_CONFIG = "config.yml"
DATA_CONFIG = "utils/data_config.yml"
pipe = CatBoostPipeline(config_path=MODEL_CONFIG)
pipe.load()

# ---- JSON schema ----

class InputJSON(BaseModel):
    age: int = Field(..., example=48)
    class_of_worker: str = Field(..., example="Private")
    education: str = Field(..., example="Masters")
    wage_per_hour: float = Field(..., example=60)
    marital_stat: str = Field(..., example="Married-civilian spouse present")
    major_industry_code: str = Field(..., example="Professional Services")
    major_occupation_code: str = Field(..., example="Executive")
    race: str = Field(..., example="White")
    hispanic_origin: str = Field(..., example="All other")
    sex: str = Field(..., example="Male")
    full_or_part_time_stat: str = Field(..., example="Full-time schedules")
    capital_gains: float = Field(..., example=10000)
    capital_losses: float = Field(..., example=0)
    dividends_from_stocks: float = Field(..., example=2000)
    tax_filer_stat: str = Field(..., example="Joint both under 65")
    household_summary: str = Field(..., example="Householder")
    live_in_this_house_1_year_ago: str = Field(..., example="Yes")
    num_persons_worked_for_employer: float = Field(..., example=2)
    family_members_under_18: str = Field(..., example="Children present")
    country_birth_father: str = Field(..., example="United-States")
    country_birth_mother: str = Field(..., example="United-States")
    country_birth_self: str = Field(..., example="United-States")
    citizenship: str = Field(..., example="Native- Born in the United States")
    own_business_or_self_employed: str = Field(..., example="No")
    veterans_benefits: str = Field(..., example="0")
    weeks_worked_in_year: float = Field(..., example=52)
    year: int = Field(..., example=1994)

class BatchInput(BaseModel):
    data: List[InputJSON]

# ---- Endpoints ----

@app.get("/health", summary="API health")
def health():
    return {"status": "ok"}

@app.get("/metadata", summary="Model and feature info")
def metadata():
    return {
        "model": "CatBoost",
        "features": pipe.feature_names,
        "cat_features": pipe.cat_features
    }

@app.post("/predict_json", summary="Predict for a single JSON object")
def predict_json(input: InputJSON):
    """Predict class and probability for a single JSON input."""
    try:
        df = pd.DataFrame([input.dict()])
        df = df[pipe.feature_names]
        pred = int(pipe.predict(df)[0])
        proba = float(pipe.predict_proba(df)[0])
        return {
            "prediction": pred,
            "proba": proba
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")

@app.post("/predict_json_batch", summary="Predict for a batch of JSON objects")
def predict_json_batch(inputs: BatchInput):
    """Predict classes and probabilities for a list of JSON objects."""
    try:
        df = pd.DataFrame([item.dict() for item in inputs.data])
        df = df[pipe.feature_names]
        preds = pipe.predict(df)
        probas = pipe.predict_proba(df)
        return {
            "predictions": preds.tolist(),
            "probas": probas.tolist()
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Batch prediction failed: {str(e)}")

@app.post("/predict_file", summary="Batch prediction from uploaded CSV file")
def predict_file(file: UploadFile = File(...)):
    """
    Predict for all rows in a CSV file (must have the same columns as the model expects).
    """
    try:
        df = pd.read_csv(file.file)
        with open(DATA_CONFIG, "r") as f:
            data_config = yaml.safe_load(f)
        X_pred, _, _, _, _ = preprocess_data(df, None, config=data_config, verbose=False)
        X_pred = X_pred[pipe.feature_names]
        preds = pipe.predict(X_pred)
        return {
            "predictions": preds.tolist(),
            "n_predictions": len(preds)
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"File prediction failed: {str(e)}")

@app.post("/test_file", summary="Evaluate model on labeled batch CSV")
def test_file(file: UploadFile = File(...)):
    """
    Evaluate model performance on a labeled CSV file (must contain the true labels).
    Returns a classification report.
    """
    try:
        df = pd.read_csv(file.file)
        with open(DATA_CONFIG, "r") as f:
            data_config = yaml.safe_load(f)
        X_pred, _, y_true, _, _ = preprocess_data(df, None, config=data_config, verbose=False)
        X_pred = X_pred[pipe.feature_names]
        preds = pipe.predict(X_pred)
        report = classification_report(y_true, preds, output_dict=True)
        return {"classification_report": report}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Test failed: {str(e)}")

@app.get("/", summary="Root")
def root():
    return {"message": "CatBoost Model API. See /docs for schema and usage."}
