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

    df_display = df.copy()

    df_display["Sex"] = df_display["Sex"].replace({"female": "Femme", "male": "Homme"})
    df_display["Survived"] = df_display["Survived"].replace({0: "Non", 1: "Oui"})

    palette = sns.color_palette("RdYlGn", n_colors=3)  # rouge - jaune - vert
    palette = [palette[2], palette[0]]  # vert et rouge

    st.write("### Analyse univariée")
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    sns.countplot(
        x="Survived", data=df_display, order=["Oui", "Non"], palette=palette, ax=axs[0]
    )
    axs[0].set_xlabel("Survie")
    for ax in axs:
        ax.set_ylabel("Nombre de passagers")
    axs[0].set_title("Survie des passagers (cible de l'étude)")

    sns.histplot(
        data=df_display,
        x="Age",
        # rug=True,
        bins=[0, 10, 20, 30, 40, 50, 60, 70, 81],
        ax=axs[1],
    )
    axs[1].set_title("Distribution de l'âge des passagers")

    sns.histplot(
        data=df_display,
        x="Fare",
        # bins=[0, 100, 200, 30, 40, 50, 60, 70, 80],
        ax=axs[2],
    )
    axs[2].set_xlabel("Tarif")
    axs[2].set_title("Distribution des tarifs")

    plt.tight_layout()

    st.pyplot(fig)
    ####################################################

    st.write(
        "Trois observations présentent un tarif de 512,33, nettement supérieur à la distribution générale. Bien que ces valeurs ne soient pas nécessairement aberrantes, elles sont considérées comme des outliers extrêmes. Afin d’éviter qu’elles ne biaisent les analyses ultérieures, elles sont exclues du jeu de données. L’analyse est ainsi restreinte aux 888 passagers ayant un tarif compris entre 0 et 263."
    )
    st.write("### Analyse bivariée")

    df_display = df_display[df_display["Fare"] < 500]

    fig, axs = plt.subplots(3, 2, figsize=(12, 12))  # 3 lignes, 2 colonnes

    sns.kdeplot(
        data=df_display,
        x="Age",
        hue="Survived",
        hue_order=["Oui", "Non"],
        palette=palette,
        cut=0,
        fill=True,
        alpha=0.6,
        ax=axs[0, 0],
    )
    axs[0, 0].set_xlabel("Âge")
    axs[0, 0].set_ylabel("Densité")
    axs[0, 0].set_title("Survie en fonction de l'âge")

    sns.histplot(
        data=df_display,
        x="Age",
        bins=17,
        hue="Survived",
        hue_order=["Non", "Oui"],
        multiple="fill",
        cumulative=False,
        palette=palette[::-1],
        fill=True,
        alpha=0.6,
        ax=axs[1, 0],
    )
    axs[1, 0].set_xlabel("Âge")
    axs[1, 0].set_ylabel("Densité")
    axs[1, 0].set_title("Survie en fonction de l'âge")

    sns.histplot(
        data=df_display,
        x="Fare",
        bins=20,
        hue="Survived",
        hue_order=["Oui", "Non"],
        multiple="fill",
        cumulative=False,
        palette=palette,
        fill=True,
        alpha=0.6,
        ax=axs[1, 1],
    )
    # axs[1, 0].set_xlim(0, 300)
    # axs[1, 1].set_ylim(0, 150)

    ###################################################

    sns.kdeplot(
        data=df_display,
        x="Fare",
        hue="Survived",
        hue_order=["Oui", "Non"],
        palette=palette,
        cut=0,
        fill=True,
        alpha=0.6,
        ax=axs[0, 1],
    )
    # axs[0, 1].set_xlim(0, 150)
    axs[0, 1].set_xlabel("Tarif")
    axs[0, 1].set_ylabel("Densité")
    axs[0, 1].set_title("Survie en fonction du tarif")

    sns.countplot(
        x="Sex",
        data=df_display,
        hue="Survived",
        order=["Femme", "Homme"],
        hue_order=["Oui", "Non"],
        palette=palette,
        # stat="count",
        # saturation=0.75,
    )

    sns.countplot(
        x="Pclass",
        data=df_display,
        hue="Survived",
        hue_order=["Oui", "Non"],
        palette=palette,
    )

    sns.countplot(
        x="SibSp",
        data=df_display,
        hue="Survived",
        hue_order=["Oui", "Non"],
        palette=palette,
    )

    sns.countplot(
        x="Parch",
        data=df_display,
        hue="Survived",
        hue_order=["Oui", "Non"],
        palette=palette,
    )
    plt.title("Survie en fonction du nombre de parents")
    plt.tight_layout()
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
