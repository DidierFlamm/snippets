# -*- coding: utf-8 -*-
import os
import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import warnings
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.utils import all_estimators
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# st.title("Titanic")

# chemin absolu vers le dossier où se trouve le script
dir_path = os.path.dirname(os.path.realpath(__file__))
img_path = os.path.join(dir_path, "titanic.png")

# url des données au format .csv
csv_url = "https://raw.githubusercontent.com/datasciencedojo/datasets/refs/heads/master/titanic.csv"


@st.cache_data
def load_csv(csv):
    df = pd.read_csv(csv, index_col="PassengerId")
    df.index.name = "Id"

@st.cache_data
def preprocess_data(df):
    # features
    X = df.copy()

    X = X.drop(
        ["Name", "Ticket", "Cabin"],
        axis=1,
    )

    # feature engineering
    X["Family"] = X["SibSp"] + X["Parch"] + 1
    X["IsAlone"] = (X["Family"] == 1).astype(int)

    # target
    y = X.pop("Survived")

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # gestion des valeurs manquantes
    age_median = X_train["Age"].median()
    embarked_mode = X_train["Embarked"].mode()[0]

    X_train["Age"] = X_train["Age"].fillna(age_median)
    X_train["Embarked"] = X_train["Embarked"].fillna(embarked_mode)

    X_test["Age"] = X_test["Age"].fillna(age_median)
    X_test["Embarked"] = X_test["Embarked"].fillna(embarked_mode)

    # scaling des variables numériques
    num_cols = ["Age", "Fare", "SibSp", "Parch", "Pclass", "Family"]
    scaler = StandardScaler()
    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test[num_cols] = scaler.transform(X_test[num_cols])

    # encodage des variables catégorielles
    categorical_cols = ["Sex", "Embarked"]
    X_train = pd.get_dummies(X_train, columns=categorical_cols, drop_first=True)
    X_test = pd.get_dummies(X_test, columns=categorical_cols, drop_first=True)
    # Réindexation pour garantir le même ordre des colonnes (pas garanti apres oh encodage)
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

    return df, X_train, X_test, y_train, y_test


###################################################################################### sidebar
st.sidebar.title("Sommaire")


pages = [
    "Accueil",
    "Visualisations",
    "Prédictions",
    "Pre-processing + Evaluation",
    "Tuning",
]

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

    df = load_csv(csv_url)
    st.dataframe(df)
    st.caption("Les valeurs grises indiquent des données manquantes.")

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

########################################################################################################################
elif page == pages[3]:

    X_train, X_test, y_train, y_test = preprocess_data(df)

    # st.dataframe(X_train)
    # st.dataframe(X_test)

    # Récupérer tous les classifieurs
    all_classifiers = all_estimators(type_filter="classifier")

    # warnings.filterwarnings("ignore")

    results = []

    progress_bar = st.progress(0)
    status = st.empty()
    total = len(all_classifiers)

    placeholder=st.empty()

    for i, (name, ClfClass) in enumerate(all_classifiers):

        progress_bar.progress((i + 1) / total)
        status.text(f"{i+1}/{total} - {name}")

        try:
            clf = ClfClass()
            start_time = time.time()
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            end_time = time.time()
            duration = int((end_time - start_time) * 1000)

            acc = accuracy_score(y_test, y_pred)
            bal_acc = balanced_accuracy_score(y_test, y_pred)
            results.append(
                {
                    "Model": name,
                    "Accuracy": acc,
                    "Balanced Accuracy": bal_acc,
                    "Time (ms)": duration,
                }
            )
        except Exception:
            results.append(
                {
                    "Model": name,
                    "Accuracy": None,
                    "Balanced Accuracy": None,
                    "Time (ms)": None,
                }
            )

        

    # Afficher sous forme de DataFrame triée par Accuracy décroissante
        df_results = pd.DataFrame(results)
        df_results = (
        df_results.dropna()
        .sort_values(by="Balanced Accuracy", ascending=False)
        .reset_index(drop=True)
    )

        placeholder.dataframe(df_results)


#######################################################################################################
elif page == pages[4]:

    models = {
        "GradientBoosting": GradientBoostingClassifier(random_state=42),
        "SVC": SVC(),
        "KNeighbors": KNeighborsClassifier(),
        "RandomForest": RandomForestClassifier(random_state=42),
    }

    params = {
        "GradientBoosting": {
            "n_estimators": [50, 100],
            "learning_rate": [0.01, 0.1],
            "max_depth": [3, 5],
        },
        "SVC": {
            "C": [0.1, 1, 10],
            "kernel": ["linear", "rbf"],
            "gamma": ["scale", "auto"],
        },
        "KNeighbors": {
            "n_neighbors": [3, 5, 7],
            "weights": ["uniform", "distance"],
        },
        "RandomForest": {
            "n_estimators": [50, 100],
            "max_depth": [None, 5, 10],
            "min_samples_split": [2, 5],
        },
    }

    best_models = {}
    results = []

    for name in models:
        print(f"🔍 GridSearch for {name}...")
        grid = GridSearchCV(
            models[name], params[name], cv=5, n_jobs=-1, scoring="balanced_accuracy"
        )
        grid.fit(X_train, y_train)

        best_model = grid.best_estimator_
        best_models[name] = best_model
        y_pred = best_model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        bal_acc = balanced_accuracy_score(y_test, y_pred)

        st.write(f"✅ {name} - Best Params: {grid.best_params_}")
        st.write(f"➡️ Accuracy: {acc:.4f}, Balanced Accuracy: {bal_acc:.4f}\n")

        results.append(
            {
                "Model": name,
                "Accuracy": acc,
                "Balanced Accuracy": bal_acc,
                "Best Params": grid.best_params_,
            }
        )

        st.dataframe(
            pd.DataFrame(grid.cv_results_).sort_values(
                "mean_test_score", ascending=False
            )
        )

    df_results = (
        pd.DataFrame(results)
        .sort_values(by="Balanced Accuracy", ascending=False)
        .reset_index(drop=True)
    )
    st.dataframe(df_results)

    knn_default = KNeighborsClassifier()
    knn_default.fit(X_train, y_train)
    y_pred_default = knn_default.predict(X_test)
    st.write("Default Accuracy:", accuracy_score(y_test, y_pred_default))
    st.write(
        "Default Balanced Accuracy:", balanced_accuracy_score(y_test, y_pred_default)
    )

    best_knn = KNeighborsClassifier(7, weights="uniform")
    best_knn.fit(X_train, y_train)
    y_pred_best = best_knn.predict(X_test)
    st.write("Best Accuracy:", accuracy_score(y_test, y_pred_best))
    st.write("Best Balanced Accuracy:", balanced_accuracy_score(y_test, y_pred_best))
