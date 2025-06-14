# -*- coding: utf-8 -*-
import os
import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

# st.title("Titanic")

# chemin absolu vers le dossier où se trouve le script
dir_path = os.path.dirname(os.path.realpath(__file__))
img_path = os.path.join(dir_path, "titanic.png")

# url des données au format .csv
raw_csv = "https://raw.githubusercontent.com/datasciencedojo/datasets/refs/heads/master/titanic.csv"
df = pd.read_csv(raw_csv, index_col="PassengerId")
df.index.name = "Id"

###################################################################################### sidebar
st.sidebar.title("Sommaire")


pages = ["Accueil", "Visualisations", "Prédictions"]

page = st.sidebar.radio("Aller vers", pages)

###################################################################################### page 0
if page == pages[0]:

    # st.write("# Introduction")

    st.image(img_path)
    st.write("")
    st.markdown(
        "Prédiction de la survie des passagers du Titanic à partir des [données](https://github.com/datasciencedojo/datasets/blob/master/titanic.csv) d'un échantillon de 891 passagers"
    )
    st.write("")
    st.dataframe(df)

    if st.checkbox("Afficher le nombre de valeurs manquantes"):
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

###################################################################################### page 1
elif page == pages[1]:
    st.write("### Analyse univariée (Seaborn)")

    fig = plt.figure()
    sns.countplot(x="Survived", data=df)
    st.pyplot(fig)

    fig = plt.figure()
    sns.countplot(x="Pclass", data=df)
    st.pyplot(fig)

    st.write("### Analyse multivariée interactive (Plotly)")

    hist = px.histogram(df, x="Survived", color="Sex", barmode="group")
    st.plotly_chart(hist)

    hist_bis = px.sunburst(df, path=["Sex", "Pclass"])
    st.plotly_chart(hist_bis)

###################################################################################### page 2
elif page == pages[2]:

    st.write("### Evaluation de la performance")

    df = df.dropna()

    df = df.drop(["Name", "Sex", "Ticket", "Cabin", "Embarked"], axis=1)

    X = df.drop("Survived", axis=1)
    y = df["Survived"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model_choisi = st.selectbox(
        label="Choix du modèle", options=["Regression Log", "Decision Tree", "KNN"]
    )

    def train_model(model_choisi):
        if model_choisi == "Regression Log":
            model = LogisticRegression()
        elif model_choisi == "Decision Tree":
            model = DecisionTreeClassifier()
        elif model_choisi == "KNN":
            model = KNeighborsClassifier()
        model.fit(X_train, y_train)  # type: ignore
        score = model.score(X_test, y_test)  # type: ignore
        return score

    score = train_model(model_choisi)
    st.write(f"Accuracy : {score:.1%}")
    st.write(
        "ℹ️ Accuracy, ou Exactitude en français, c'est le taux de prédictions justes du modèle"
    )

    st.write("### Prédictions")

    code = """🚧 Bientôt la possibilité d'évaluer la probabilité de survie\nen fonction de vos paramètres 🚧"""
    st.code(code, language="python")
