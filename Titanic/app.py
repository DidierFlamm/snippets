# -*- coding: utf-8 -*-
import os
import random

import streamlit as st
from utils import load_csv

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
from sklearn.utils import all_estimators
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression


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


# chemin absolu vers le png
dir_path = os.path.dirname(os.path.realpath(__file__))
img_path = os.path.join(dir_path, "assets/titanic.png")


# mise en cache de la seed
if "seed" not in st.session_state:
    st.session_state.seed = random.randint(0, 2**32 - 1)

seed = st.session_state.seed

# fixer la seed de toutes les fonctions faisant appel à random_state
random.seed(seed)

st.set_page_config(
    page_title="Accueil"
)  # facultatif car le script s'appelle Accueil.py


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

###################################################################################### Evaluation
if page == pages[2]:

    st.header("Evaluation")

    # Récupérer tous les classifieurs
    all_classifiers = all_estimators(type_filter="classifier")

    st.write(
        "Entraînement puis évaluation de la performance par stratified-KFold Cross Validation des modèles de la librairie scikit-learn"
    )

    df = load_csv(csv_path)

    X_train, X_test, y_train, y_test = preprocess_data(df)

    # st.dataframe(X_train)
    # st.dataframe(X_test)

    # Récupérer tous les classifieurs
    all_classifiers = all_estimators(type_filter="classifier")

    # warnings.filterwarnings("ignore")

    results = []
    errors = []
    df_results = pd.DataFrame()

    progress_bar = st.progress(0)
    status = st.empty()
    total = len(all_classifiers)

    placeholder = st.empty()

    st.caption(
        f"Evaluation réalisée par cross validation avec une seed fixée aléatoirement = {seed}"
    )

    start_total_time = time.time()

    skf = StratifiedKFold(n_splits=5, shuffle=True)

    for i, (name, ClfClass) in enumerate(all_classifiers):

        progress_bar.progress((i + 1) / total)
        status.text(f"{i+1}/{total} - {name}")

        try:
            clf = ClfClass()
            start_time = time.time()

            bal_acc_scores = cross_val_score(
                clf, X_train, y_train, cv=skf, scoring="balanced_accuracy"
            )

            roc_auc_scores = f1_scores = cross_val_score(
                clf, X_train, y_train, cv=skf, scoring="roc_auc"
            )

            f1_scores = cross_val_score(clf, X_train, y_train, cv=skf, scoring="f1")

            bal_acc_mean = bal_acc_scores.mean()
            roc_auc_mean = roc_auc_scores.mean()
            f1_mean = f1_scores.mean()

            end_time = time.time()
            duration = int((end_time - start_time) * 1000)

            if pd.isna(bal_acc_mean) or pd.isna(roc_auc_mean) or pd.isna(f1_mean):
                raise ValueError("Scores invalides (nan)")

            results.append(
                {
                    "Model": name,
                    "Balanced Accuracy (%)": round(100 * bal_acc_mean, 2),
                    "ROC AUC": roc_auc_mean,
                    "f1-score": f1_mean,
                    "Time (ms)": duration,
                }
            )
        except Exception as e:
            errors.append({"Model": name, "Error": e})

        # Afficher sous forme de DataFrame triée par Accuracy décroissante
        df_results = pd.DataFrame(results)
        df_results = df_results.sort_values(
            by="Balanced Accuracy (%)", ascending=False
        ).reset_index(drop=True)

        placeholder.dataframe(df_results)

    duration = int(1000 * (time.time() - start_total_time))
    status.text(
        f"ℹ️ {len(all_classifiers)} modèles évalués en {duration} ms (✔️ {len(results)} succès, ❌ {len(errors)} erreurs)"
    )

    with st.expander("Afficher les erreurs"):
        st.dataframe(errors)

    st.markdown("---")

    best_model_name = df_results.iloc[0, 0]

    st.write(
        f"🥇 {best_model_name} présente la balanced accuracy la plus élevée : {df_results.iloc[0, 1]} %"
    )

    best_model = None

    for name, Clf in all_classifiers:
        if name == best_model_name:
            best_model = Clf()
            break
    else:
        raise ValueError(
            f"Impossible de trouver {best_model_name} dans all_classifiers"
        )

    assert best_model is not None, f"best_model_name {best_model_name} non trouvé"

    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)

    # Afficher classification_report sous forme de DataFrame
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    df_report = pd.DataFrame(report_dict).transpose()
    st.markdown("- Classification Report")
    st.dataframe(df_report)

    # Afficher la matrice de confusion
    cm = confusion_matrix(y_test, y_pred)
    df_cm = pd.DataFrame(
        cm, index=["Actual 0", "Actual 1"], columns=["Pred 0", "Pred 1"]
    )
    st.markdown("- Classification Report")
    st.dataframe(df_cm)


###################################################################################### Optimisation
elif page == pages[3]:
    st.header("Optimisation")
    st.subheader("🔧 Fine tuning des hyperparamètres de 5 modèles")

    df = load_csv(csv_path)

    X_train, X_test, y_train, y_test = preprocess_data(df)

    models = {
        "LogisticRegression": LogisticRegression(),
        "KNeighbors": KNeighborsClassifier(),
        "SVC": SVC(probability=True),
        "RandomForest": RandomForestClassifier(),
        "GradientBoosting": GradientBoostingClassifier(),
    }

    params = {
        "LogisticRegression": {
            "C": [0.01, 0.1, 1, 10],
            "penalty": ["l2"],
            "solver": ["lbfgs"],
        },
        "KNeighbors": {
            "n_neighbors": [3, 5, 7],
            "weights": ["uniform", "distance"],
        },
        "SVC": {
            "C": [0.1, 1, 10],
            "kernel": ["linear", "rbf"],
            "gamma": ["scale", "auto"],
        },
        "RandomForest": {
            "n_estimators": [50, 100],
            "max_depth": [None, 5, 10],
            "min_samples_split": [2, 5],
        },
        "GradientBoosting": {
            "n_estimators": [50, 100],
            "learning_rate": [0.01, 0.1],
            "max_depth": [3, 5],
        },
    }

    best_models = {}
    results = []

    for name in models:
        # print(f"🔍 GridSearch for {name}...")
        grid = GridSearchCV(
            models[name], params[name], cv=5, n_jobs=-1, scoring="balanced_accuracy"
        )
        grid.fit(X_train, y_train)

        best_model = grid.best_estimator_
        best_models[name] = best_model
        y_pred = best_model.predict(X_test)

        bal_acc = balanced_accuracy_score(y_test, y_pred)

        st.markdown(
            f"""
    - **{name}**  
        Best Params : {grid.best_params_}  
        Balanced Accuracy : **{bal_acc:.4f}**  
    """
        )

        results.append(
            {
                "Model": name,
                "Balanced Accuracy": bal_acc,
                "Best Params": grid.best_params_,
            }
        )
        with st.expander("Afficher les détails"):
            st.dataframe(pd.DataFrame(grid.cv_results_))

        st.markdown("---")

    st.subheader("🎯 Résultats")

    df_results = (
        pd.DataFrame(results)
        .sort_values(by="Balanced Accuracy", ascending=False)
        .reset_index(drop=True)
    )
    st.dataframe(df_results)

elif page == pages[4]:

    st.header("Prédictions")

    model_choisi = st.selectbox(
        label="Choix du modèle", options=["Regression Log", "Decision Tree", "KNN"]
    )
