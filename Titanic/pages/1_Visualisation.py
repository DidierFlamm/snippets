import streamlit as st
from utils import load_csv
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

st.set_page_config(page_title="Titanic")
st.header("Visualisation")
st.sidebar.write("© 2025 Didier Flamm")

df = load_csv()

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
