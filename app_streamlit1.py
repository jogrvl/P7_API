# app_streamlit.py

import streamlit as st
import requests

st.set_page_config(page_title="Scoring Crédit - Test API", page_icon="💳")

st.title("💳 Test de l'API Scoring Crédit P7")
st.write("Entrez les données du client pour obtenir une prédiction :")

# --- Saisie des données ---
DAYS_EMPLOYED = st.number_input("Jours d'emploi (DAYS_EMPLOYED)", value=-2000.0)
AMT_INCOME_TOTAL = st.number_input("Revenu total (AMT_INCOME_TOTAL)", value=150000.0)
AMT_CREDIT = st.number_input("Montant du crédit (AMT_CREDIT)", value=600000.0)
APPROVED_DECISION_MAX = st.number_input("Score max décision précédente (APPROVED_DECISION_MAX)", value=0.0)

# URL de ton API en ligne
api_url = "https://p7-api-2m7n.onrender.com/predict"

if st.button("🔮 Lancer la prédiction"):
    data = {
        "DAYS_EMPLOYED": DAYS_EMPLOYED,
        "AMT_INCOME_TOTAL": AMT_INCOME_TOTAL,
        "AMT_CREDIT": AMT_CREDIT,
        "APPROVED_DECISION_MAX": APPROVED_DECISION_MAX
    }

    st.write("📤 Envoi des données :", data)

    try:
        response = requests.post(api_url, json=data)
        if response.status_code == 200:
            result = response.json()
            st.success(f"Résultat : {result['prediction']}")
            st.metric("Score de probabilité", result["score_probabilite"])
            st.info(f"Seuil utilisé : {result['seuil_utilise']}")
        else:
            st.error(f"Erreur API : {response.status_code}")
    except Exception as e:
        st.error(f"❌ Erreur de connexion : {e}")
