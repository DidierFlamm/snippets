import os
import streamlit as st
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


dir_path = os.path.dirname(os.path.realpath(__file__))
csv_path = os.path.join(dir_path, "data/titanic.csv")


def set_seed():
    if "seed" not in st.session_state:
        st.session_state.seed = random.randint(0, 2**32 - 1)
    seed = st.session_state.seed
    random.seed(seed)
    np.random.seed(seed)


@st.cache_data
def load_csv():
    df = pd.read_csv(csv_path, index_col="PassengerId")
    df.index.name = "Id"
    return df


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
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

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

    return X_train, X_test, y_train, y_test
