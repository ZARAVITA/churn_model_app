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
    if Geography.lower() == "france":
        geography_encoded = [1, 0, 0]
    elif Geography.lower() == "spain":
        geography_encoded = [0, 1, 0]
    elif Geography.lower() == "germany":
        geography_encoded = [0, 0, 1]
    input_data = np.array([[*geography_encoded, CreditScore, gender_encoded, Age, Tenure, Balance, 
                         NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary]])
    input_data = sc.transform(input_data)
    prediction = model.predict(input_data)[0,0]
    return prediction

# Interface Streamlit
st.set_page_config(page_title="Prédiction de Churn Bancaire", layout="wide")

# Ajouter le logo en haut à droite
col1, col2 = st.columns([10, 2])
with col1:
    st.title("Prédiction de Churn Bancaire")
with col2:
    st.image("DataPowerZLogo.png", width=290)

# Formulaire pour saisir les informations du client
with st.form("client_info"):
    st.header("Informations du client")
    Geography = st.selectbox("Region", ["France", "Spain", "Germany"])
    CreditScore = st.slider("Score de crédit", 0, 850, 600)
    Gender = st.radio("Genre", ["Male", "Female"])
    Age = st.slider("Âge", 18, 100, 40)
    Tenure = st.slider("Ancienneté (années)", 0, 40, 3)
    Balance = st.number_input("Balance(restant dans le compte)", value=60000)
    NumOfProducts = st.slider("Nombre de produits", 1, 4, 2)
    HasCrCard = st.radio("Carte de crédit", [1, 0])
    IsActiveMember = st.radio("Membre actif(1=actif)", [1, 0])
    EstimatedSalary = st.number_input("Salaire estimé par an", value=50000)
    submitted = st.form_submit_button("Prédire")

if submitted:
    prediction = predict_churn(Geography, CreditScore, Gender, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary)
    stay_prob = (1 - prediction) * 100
    leave_prob = prediction * 100

    # Créer un graphique circulaire
    fig, ax = plt.subplots(figsize=(0.5,0.5))
    ax.pie([stay_prob, leave_prob], labels=[f"Rester ({stay_prob:.2f}%)", f"Quitter ({leave_prob:.2f}%)"], 
           colors=["green", "red"], 
           autopct='%1.1f%%',textprops={'fontsize': 10},  # Taille du texte réduite
           pctdistance=0.8,  # Distance des pourcentages par rapport au centre
           labeldistance=2.1  # Distance des étiquettes par rapport au centre
    )
    ax.axis('equal')  # Assure que le pie chart est un cercle.

    st.pyplot(fig)

    if prediction > 0.5:
        st.error(f"🚨 Le client a {leave_prob:.2f}% de chances de quitter la banque.")
    else:
        st.success(f"✅ Le client a {stay_prob:.2f}% de chances de rester.")
