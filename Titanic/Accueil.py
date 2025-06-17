# -*- coding: utf-8 -*-
import os
import random

import streamlit as st
from utils import set_seed, load_csv

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

# import warnings
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score



# Options de texte possibles sur streamlit :

# Une ligne vide pour espacer un peu :

# st.write("")

# Séparateur (trait horizontal)

# st.markdown("---")

# Titre de l'app
# st.title("Titre")
# st.sidebar.title("Titre du side")

# Un titre ou sous-titre :

# st.header("Titre de section")
# st.subheader("Sous-titre")

# Ou du texte en italique ou gras :

# st.markdown("*Texte en italique*")
# st.markdown("**Texte en gras**")

# fige la seed de la session streamlit
set_seed()

# chemin absolu vers le png
dir_path = os.path.dirname(os.path.realpath(__file__))
img_path = os.path.join(dir_path, "assets/titanic.png")

st.set_page_config(page_title="Accueil")
# facultatif car le script s'appelle Accueil.py


st.image(img_path)
st.write("")
st.markdown(
    "Prédiction de la survie des passagers du Titanic à partir des [données](https://github.com/datasciencedojo/datasets/blob/master/titanic.csv) d'un échantillon de 891 passagers"
)
st.write("")

df = load_csv()
st.dataframe(df)
st.caption("Les valeurs grises indiquent des données manquantes.")

with st.expander("Afficher les valeurs manquantes"):
    # Compter les valeurs manquantes et formater proprement
    missing = df.isna().sum().to_frame(name="Valeurs manquantes")
    missing["%"] = missing["Valeurs manquantes"] / len(df)
    missing["%"] = missing["%"].map(lambda x: f"{x:.1%}")
    # filtre et trie des valeurs manquantes
    missing = missing[missing["Valeurs manquantes"] > 0]
    missing = missing.sort_values("Valeurs manquantes", ascending=False)
    # affiche en markdown pour avoir style center
    st.markdown(
        missing.style.set_properties(**{"text-align": "center"}).to_html(),  # type: ignore
        unsafe_allow_html=True,
    )

page = 1
pages = [
    0,
    0,
    0,
    0,
    0,
]




elif page == pages[4]:

    st.header("Prédictions")

    model_choisi = st.selectbox(
        label="Choix du modèle", options=["Regression Log", "Decision Tree", "KNN"]
    )
