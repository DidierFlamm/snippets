import streamlit as st
from utils import set_seed


st.set_page_config(page_title="Titanic - Predictions")
st.header("Prédictions")


set_seed()

model_choisi = st.selectbox(
    label="Choix du modèle", options=["Regression Log", "Decision Tree", "KNN"]
)


st.markdown(
    """
    <div style='text-align: center; font-size: small; color: gray; margin-top: 50px;'>
    © 2025 Didier Flamm
    </div>
    """,
    unsafe_allow_html=True,
)
