# src/api.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import os
import requests

# -----------------------------
# Paramètres
# -----------------------------
THRESHOLD_METIER = 0.54
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "train_df_cleaned.csv")
DRIVE_DOWNLOAD_URL = "https://drive.google.com/uc?export=download&id=1pOgCnUZYEmmhevjbj2jBQU2Tre0uhUG6"
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "modele_pipeline.pkl")

# -----------------------------
# Télécharger le CSV si nécessaire
# -----------------------------
if not os.path.exists(DATA_PATH):
    print("Téléchargement de train_df_cleaned.csv depuis Google Drive...")
    response = requests.get(DRIVE_DOWNLOAD_URL, stream=True)
    response.raise_for_status()
    with open(DATA_PATH, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    print("Téléchargement terminé.")

# Charger le CSV
df_clients = pd.read_csv(DATA_PATH)
df_clients.set_index("SK_ID_CURR", inplace=True)  # Utiliser SK_ID_CURR comme index

# -----------------------------
# Charger le pipeline
# -----------------------------
pipe = joblib.load(MODEL_PATH)
ALL_COLUMNS = pipe.feature_names_in_  # sklearn >=1.0

# -----------------------------
# Initialisation FastAPI
# -----------------------------
app = FastAPI(title="API Scoring Crédit P7", version="1.0")

# -----------------------------
# Modèle de données pour l'entrée
# -----------------------------
class ClientRequest(BaseModel):
    SK_ID_CURR: int

# -----------------------------
# Routes
# -----------------------------
@app.get("/")
def root():
    return {"message": "API Scoring Crédit - OK"}

@app.post("/predict")
def predict_score(request: ClientRequest):
    client_id = request.SK_ID_CURR

    # Vérifier que l'ID existe dans le CSV
    if client_id not in df_clients.index:
        raise HTTPException(status_code=404, detail=f"Client {client_id} non trouvé")

    # Récupérer les données du client
    client_data = df_clients.loc[client_id].to_dict()

    # Compléter les colonnes manquantes du pipeline
    full_dict = {col: 0.0 for col in ALL_COLUMNS}
    for col in client_data:
        if col in ALL_COLUMNS:
            full_dict[col] = client_data[col]

    # DataFrame pour la prédiction
    df = pd.DataFrame([full_dict])

    # Prédiction
    proba = float(pipe.predict_proba(df)[0][1])
    prediction = int(proba > THRESHOLD_METIER)

    return {
        "score_probabilite": round(proba, 4),
        "prediction": "Refusé" if prediction == 1 else "Approuvé",
        "seuil_utilise": THRESHOLD_METIER
    }
