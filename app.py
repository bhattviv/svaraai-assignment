from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import joblib
import numpy as np


vectorizer = joblib.load("tfidf_vectorizer.pkl")
log_reg = joblib.load("baseline_logreg_model.pkl")
lgbm = joblib.load("baseline_lgbm_model.pkl")
label_mapping = joblib.load("label_mapping.pkl")
reverse_mapping = {v: k.lower() for k, v in label_mapping.items()}  # standardized labels


class TextInput(BaseModel):
    texts: List[str]


app = FastAPI(title="Sentiment Analysis API")

@app.get("/")
def home():
    return {"message": "Sentiment Analysis API is running!"}

@app.post("/predict")
def predict_sentiment(input: TextInput):
    texts = input.texts
    X = vectorizer.transform(texts)
    
    # Logistic Regression predictions
    pred_log = log_reg.predict(X)
    pred_log_labels = [reverse_mapping[p] for p in pred_log]

    # LightGBM predictions
    pred_lgb = lgbm.predict(X).argmax(axis=1)
    pred_lgb_labels = [reverse_mapping[p] for p in pred_lgb]

    # Return results
    results = []
    for i, text in enumerate(texts):
        results.append({
            "text": text,
            "logistic_regression": pred_log_labels[i],
            "lightgbm": pred_lgb_labels[i]
        })

    return {"predictions": results}
