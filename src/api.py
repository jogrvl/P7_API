# src/api.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import os

# Seuil métier optimal
THRESHOLD_METIER = 0.54

# Chemin vers le pipeline et les données
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "..", "modele_pipeline.pkl")
DATA_PATH = os.path.join(BASE_DIR, "..", "train_df_cleaned.csv")

# Charger le pipeline et les données
pipe = joblib.load(MODEL_PATH)
df_clients = pd.read_csv(DATA_PATH)

# Colonnes attendues par le pipeline
ALL_COLUMNS = pipe.feature_names_in_

# Créer l'application FastAPI
app = FastAPI(title="API Scoring Crédit P7", version="1.0")

# Modèle Pydantic pour l'input
class ClientRequest(BaseModel):
    SK_ID_CURR: int

@app.get("/")
def root():
    return {"message": "API Scoring Crédit - OK"}

@app.post("/predict")
def predict_score(request: ClientRequest):
    # Chercher le client dans le DataFrame
    client_row = df_clients[df_clients["SK_ID_CURR"] == request.SK_ID_CURR]
    if client_row.empty:
        raise HTTPException(status_code=404, detail=f"Client {request.SK_ID_CURR} non trouvé.")

    # Prendre la première ligne (en cas de doublon)
    client_dict = client_row.iloc[0].to_dict()

    # Compléter toutes les colonnes manquantes par 0
    full_dict = {col: client_dict.get(col, 0.0) for col in ALL_COLUMNS}

    # DataFrame pour le pipeline
    df_input = pd.DataFrame([full_dict])

    # Prédiction
    proba = float(pipe.predict_proba(df_input)[0][1])
    prediction = int(proba > THRESHOLD_METIER)

    return {
        "score_probabilite": round(proba, 4),
        "prediction": "Refusé" if prediction == 1 else "Approuvé",
        "seuil_utilise": THRESHOLD_METIER
    }
