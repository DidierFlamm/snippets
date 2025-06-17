import streamlit as st
from utils import set_seed, load_csv, preprocess_data
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import balanced_accuracy_score
import pandas as pd

st.set_page_config(page_title="Titanic")
st.header("Optimisation")
st.subheader("🔧 Fine tuning des hyperparamètres de 5 modèles")

set_seed()
df = load_csv()

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
