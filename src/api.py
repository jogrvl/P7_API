# src/api.py

from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import os

# Seuil métier optimal
THRESHOLD_METIER = 0.54

# Charger le pipeline
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "modele_pipeline.pkl")
pipe = joblib.load(MODEL_PATH)

# Liste de toutes les colonnes du pipeline
ALL_COLUMNS = pipe.feature_names_in_  # sklearn >=1.0, sinon X_train.columns.tolist()

# Créer l'application FastAPI
app = FastAPI(title="API Scoring Crédit P7", version="1.0")

# Définir les features principales que l'utilisateur doit remplir
class ClientData(BaseModel):
    DAYS_EMPLOYED: float
    AMT_INCOME_TOTAL: float
    AMT_CREDIT: float
    APPROVED_DECISION_MAX: float
    # Ajoute ici d'autres features principales si tu veux

@app.get("/")
def root():
    return {"message": "API Scoring Crédit - OK"}

@app.post("/predict")
def predict_score(data: ClientData):
    # Convertir l'entrée utilisateur en dictionnaire
    input_dict = data.dict()
    
    # Remplir un dictionnaire avec toutes les colonnes du pipeline
    full_dict = {col: 0.0 for col in ALL_COLUMNS}  # valeurs par défaut
    for col in input_dict:
        if col in ALL_COLUMNS:
            full_dict[col] = input_dict[col]

    # Créer le DataFrame pour le pipeline
    df = pd.DataFrame([full_dict])

    # Prédiction
    proba = float(pipe.predict_proba(df)[0][1])
    prediction = int(proba > THRESHOLD_METIER)

    return {
        "score_probabilite": round(proba, 4),
        "prediction": "Refusé" if prediction == 1 else "Approuvé",
        "seuil_utilise": THRESHOLD_METIER
    }
