# tests/test_api.py
from fastapi.testclient import TestClient
from src.api import app  # ton app FastAPI

# Créer le client pour les tests
client = TestClient(app)

def test_root():
    """Test de la route racine"""
    response = client.get("/")
    assert response.status_code == 200
    assert "API Scoring Crédit" in response.json()["message"]

def test_predict():
    """Test de la route /predict avec un jeu de données factice"""
    payload = {
        "DAYS_EMPLOYED": -2000,
        "AMT_INCOME_TOTAL": 150000,
        "AMT_CREDIT": 600000,
        "APPROVED_DECISION_MAX": 0.5
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    json_resp = response.json()
    assert "score_probabilite" in json_resp
    assert "prediction" in json_resp
    assert "seuil_utilise" in json_resp
