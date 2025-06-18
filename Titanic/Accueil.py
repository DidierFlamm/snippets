# -*- coding: utf-8 -*-
import os
import streamlit as st
from utils import load_csv

# Options de texte possibles sur streamlit :

# Une ligne vide pour espacer un peu :

# st.write("")

# Séparateur (trait horizontal)

# st.markdown("---")

# Titre de l'app
# st.title("Titre")

# Un titre ou sous-titre :

# st.header("Titre de section")
# st.subheader("Sous-titre")
# st.sidebar.title("Titre de section dans le side bar")

# Ou du texte en italique ou gras :

# st.markdown("*Texte en italique*")
# st.markdown("**Texte en gras**")


st.set_page_config(page_title="Titanic")
# définit le nom de l'onglet (doit être la 1ere commande)
st.title("Titanic")

st.sidebar.write("© 2025 Didier Flamm")

# chemin absolu vers le png
dir_path = os.path.dirname(os.path.realpath(__file__))
img_path = os.path.join(dir_path, "assets/titanic.png")


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
