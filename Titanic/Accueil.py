# -*- coding: utf-8 -*-
import os
import streamlit as st
from utils import load_csv
import pandas as pd


# définit le nom de l'onglet (doit être la 1ere commande streamlit du script)
st.set_page_config(page_title="Titanic")

st.title("Titanic")

st.image("assets/titanic.webp")
st.logo("assets/logo.webp")

st.markdown(
    "Prédiction de la survie des passagers du Titanic à partir des [données](https://github.com/datasciencedojo/datasets/blob/master/titanic.csv) d'un échantillon de 891 passagers"
)
st.write("")


st.sidebar.markdown(
    """
    <div style="text-align: center;">
        <a href="https://share.streamlit.io/user/didierflamm" target="_blank">
            <img src="assets/maigal.png" width="150">
        </a>
    </div>
    """,
    unsafe_allow_html=True,
)
st.sidebar.image("assets/maigal.png", width=150)

st.sidebar.caption("© 2025 Didier Flamm")

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

st.markdown("---")
st.write("Définition des variables :")
df = pd.DataFrame(
    {
        "Variable": [
            "    Survived",
            "    Pclass",
            "    SibSp",
            "    Parch",
            "    Fare",
            "    Cabin",
            "    Embarked",
        ],
        "Définition": [
            "Survie du passager",
            "Classe du billet",
            "Nombre de frères, Sœurs, époux ou épouse à bord du Titanic",
            "Nombre de parents et enfants à bord du Titanic",
            "Tarif de la cabine (pour l'ensemble des occupants)",
            "Numéro de la cabine",
            "Port d'embarquement",
        ],
        "Valeurs": [
            "0 = Non, 1 = Oui",
            "1 = 1ère (haut de gamme, \n2 = 2ème (moyen de gamme), 3 = 3ème (bas de gamme))",
            "",
            "",
            "",
            "",
            "C = Cherbourg, Q = Queenstown, S = Southampton",
        ],
    }
)

st.table(df.set_index("Variable"))

st.image("assets/route.png")
