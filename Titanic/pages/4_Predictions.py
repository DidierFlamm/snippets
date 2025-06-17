import streamlit as st

st.set_page_config(page_title="Titanic")
st.header("Prédictions")

model_choisi = st.selectbox(
    label="Choix du modèle", options=["Regression Log", "Decision Tree", "KNN"]
)
