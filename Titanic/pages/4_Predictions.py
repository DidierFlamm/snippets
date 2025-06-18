import streamlit as st
from utils import set_seed


st.set_page_config(page_title="Titanic")
st.header("Prédictions")
st.sidebar.write("© 2025 Didier Flamm")

set_seed()

model_choisi = st.selectbox(
    label="Choix du modèle", options=["Regression Log", "Decision Tree", "KNN"]
)
