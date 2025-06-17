import streamlit as st

st.header("Prédictions")

model_choisi = st.selectbox(
    label="Choix du modèle", options=["Regression Log", "Decision Tree", "KNN"]
)
