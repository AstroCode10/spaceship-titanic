from fastapi import FastAPI
import joblib
import pandas as pd
import numpy as np
from catboost import Pool
from pydantic import BaseModel
from typing import Optional
from transformers import OutlierRemover, LogTransformer

class PassengerInput(BaseModel):
    HomePlanet: Optional[str]
    CryoSleep: Optional[bool]
    Destination: Optional[str]
    Age: Optional[float]
    VIP: Optional[bool]

    RoomService: Optional[float]
    FoodCourt: Optional[float]
    ShoppingMall: Optional[float]
    Spa: Optional[float]
    VRDeck: Optional[float]

    CabinDeck: Optional[str]
    CabinNum: Optional[float]
    CabinSide: Optional[str]

    GroupSize: Optional[int]

app = FastAPI(title="Spaceship Titanic Classifier")
artifact = joblib.load("spaceship_model.joblib")

model = artifact["model"]
num_pipeline = artifact["num_pipeline"]
cat_pipeline = artifact["cat_pipeline"]
cols = artifact["feature_columns"]
cat_indices = artifact["cat_indices"]

@app.post("/predict")
def predict(input_data: PassengerInput):
    df = pd.DataFrame([input_data.model_dump()])

    for col in df.columns:
        if df[col].dtype == 'bool':
            df[col] = df[col].astype('category')

    num_cols = df.select_dtypes(include=["int64", "float64"])
    cat_cols = df.select_dtypes(include=["object", "category"])

    df = df.reindex(columns=cols, fill_value=np.nan)
    for col in cat_cols.columns:
        df[col] = df[col].astype("category")
        df[col] = df[col].cat.add_categories("Unknown")
        df[col] = df[col].fillna("Unknown")

    for col in num_cols.columns:
        df[col] = df[col].fillna(0)

    num_cols_to_use = [c for c in num_cols.columns if c in df.columns]
    df[num_cols_to_use] = num_pipeline.transform(df[num_cols_to_use])
    df[cat_cols.columns] = cat_pipeline.transform(df[cat_cols.columns])

    pool = Pool(df, cat_features=cat_indices)
    prob = model.predict_proba(pool)[0, 1]
    pred = prob > 0.5

    return {
        "Transported Probability": float(prob),
        "Transported": bool(pred)
    }