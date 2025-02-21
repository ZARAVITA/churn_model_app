# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 16:06:35 2025

@author: ZARAVITA Haydar
"""

import numpy as np
import tensorflow as tf
import joblib
import streamlit as st
import matplotlib.pyplot as plt

# Charger le modèle et le scaler
model = tf.keras.models.load_model("churn_model.h5")
sc = joblib.load("scaler.pkl")
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Fonction pour prédire le churn
def predict_churn(Geography, CreditScore, Gender, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary):
    gender_encoded = 1 if Gender.lower() == "male" else 0
    geography_encoded = [0, 0, 0]  # [France, Spain, Germany]
    #"Nord & Centre[Tanger-Rabat-Casablanca-Fès]" "Intérieur & Est[Oujda-Nador-Béni Mellal]"  "Sud[Marrakech-Agadir-Laâyoune]"
    if Geography.lower() == "Nord & Centre[Tanger-Rabat-Casablanca-Fès]":
        geography_encoded = [1, 0, 0]
    elif Geography.lower() == "Intérieur & Est[Oujda-Nador-Béni Mellal]":
        geography_encoded = [0, 1, 0]
    elif Geography.lower() == "Sud[Marrakech-Agadir-Laâyoune]":
        geography_encoded = [0, 0, 1]
    input_data = np.array([[*geography_encoded, CreditScore, gender_encoded, Age, Tenure, Balance, 
                         NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary]])
    input_data = sc.transform(input_data)
    prediction = model.predict(input_data)[0,0]
    return prediction

# Interface Streamlit
st.set_page_config(page_title="ChurnPredictorZ", layout="wide")

# Ajouter le logo en haut à droite
col1, col2 = st.columns([10, 2])
with col1:
    st.title("ChurnPredictorZ")
    st.markdown("""
## 🔍 Prédiction de Churn Bancaire  
Anticipez le départ de vos clients grâce à un modèle avancé de **réseaux de neurones artificiels (ANN)**.  
Avec une précision de **87,8 %**, notre algorithme analyse les comportements bancaires et estime la probabilité qu’un client quitte la banque.  

### 🚀 Comment ça marche ?  
1️⃣ **Entrez les informations du client** (score de crédit, âge, solde, etc.).  
2️⃣ **Cliquez sur "Prédire"**.  
3️⃣ **Obtenez immédiatement** la probabilité qu’il reste ou quitte la banque.  

📊 **Pourquoi un modèle ANN ?**  
La régression logistique atteint 80 % de précision, mais notre ANN, grâce à sa capacité à capturer des relations complexes, améliore la fiabilité des prévisions avec **87,8 % de précision**.  

💡 **Optimisez la fidélisation client et réduisez l’attrition avec l’IA !**  

---  

**🔹 Projet réalisé par :**  **ZARAVITA**  
📧 Contact : zaravitamds18@gmail.com  
🎓 **Master en Modélisation Mathématique et Data Science**  
📍 Passionné par l’IA, la Data Science et leur application dans le secteur financier.  
""")

with col2:
    st.image("DataPowerZLogo.png", width=290)

# Formulaire pour saisir les informations du client
with st.form("client_info"):
    st.header("Informations du client")
    col1, col2, col3 = st.columns(3)
    with col1:
        Gender = st.selectbox("⚤ Genre", ["Male", "Female"])  # Utilisation de selectbox au lieu de radio
    with col2:
        HasCrCarde = st.selectbox("💳 Carte de crédit", ["Oui", "Non"], index=0)
        HasCrCard =1 if HasCrCarde=="Oui" else 0
    with col3:
        #Geography = st.selectbox("🌍 Région", ["France", "Spain", "Germany"])
        #"Nord & Centre[Tanger-Rabat-Casablanca-Fès]" "Intérieur & Est[Oujda-Nador-Béni Mellal]"  "Sud[Marrakech-Agadir-Laâyoune]"
        Geography = st.selectbox("Région", ["Nord & Centre[Tanger-Rabat-Casablanca-Fès]", "Intérieur & Est[Oujda-Nador-Béni Mellal]", "Sud[Marrakech-Agadir-Laâyoune]"])
    col3, col4, col5 = st.columns(3)
    with col3:
        Age = st.number_input("📅 Âge", min_value=18, max_value=100, value=40)
    with col4:
        Tenure = st.number_input("📆 Ancienneté (années)", min_value=0, max_value=40, value=3)
    with col5:
        NumOfProducts = st.number_input("📦 Nombre de produits utilisés", min_value=1, max_value=4, value=2)
    col6, col7, col8= st.columns(3)
    with col6:
        Balance = st.number_input("💰 Balance (restant dans le compte)", value=10000)
    with col7:
        EstimatedSalary = st.number_input("💵 Salaire estimé par an", value=100000)
    with col8:
        IsActiveMembere = st.selectbox("👥 Membre actif", ["Oui", "Non"], index=0)
        IsActiveMember =1 if IsActiveMembere=="Oui" else 0
    CreditScore = st.slider("💳 Score de crédit", 0, 850, 600)
    submitted = st.form_submit_button("🚀 Prédire")

if submitted:
    prediction = predict_churn(Geography, CreditScore, Gender, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary)
    stay_prob = (1 - prediction) * 100
    leave_prob = prediction * 100

    if prediction > 0.5:
        st.error(f"🚨PERTE IMMINENTE ! Un client à {leave_prob:.2f}% de risque de départ. Intervenez maintenant ou perdez-le définitivement !")
    else:
        st.success(f"✅ Bonne nouvelle ! Ce client a {stay_prob:.2f}% de chances de rester fidèle à votre banque")
#🚨PERTE IMMINENTE ! Un client à 85.22% de risque de départ. Intervenez maintenant ou perdez-le définitivement !
