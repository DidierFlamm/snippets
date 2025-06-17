import streamlit as st
from sklearn.utils import all_estimators
from utils import set_seed, load_csv, preprocess_data
import pandas as pd
import time
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix


st.header("Evaluation")

set_seed()

# Récupérer tous les classifieurs
all_classifiers = all_estimators(type_filter="classifier")

st.write(
    "Entraînement puis évaluation de la performance par stratified-KFold Cross Validation des modèles de la librairie scikit-learn"
)

df = load_csv()

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
    f"Evaluation réalisée par cross validation avec une seed fixée aléatoirement = {st.session_state.seed}"
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
    raise ValueError(f"Impossible de trouver {best_model_name} dans all_classifiers")

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
df_cm = pd.DataFrame(cm, index=["Actual 0", "Actual 1"], columns=["Pred 0", "Pred 1"])
st.markdown("- Confusion Matrix")
st.dataframe(df_cm)
