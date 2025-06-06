# -*- coding: utf-8 -*-
"""

"""
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

st.title("Test streamlit by Did")

# chemin absolu vers le dossier où se trouve le script
dir_path = os.path.dirname(os.path.realpath(__file__))
csv_path = os.path.join(dir_path, 'titanic.csv')
img_path = os.path.join(dir_path, 'titanic.jpg')

df = pd.read_csv(csv_path)


st.sidebar.title("Sommaire")


pages = ["Intro", "Dataviz", "Modélisation"]

page = st.sidebar.radio("Aller vers", pages)

if page == pages[0] : 
    
    st.write("### Introduction")
    
    st.image(img_path)
    
    st.dataframe(df.head())
    
    st.markdown("PrÃ©diction de la survie des passagers du [titanic](https://www.kaggle.com/datasets/brendan45774/test-file)")
    
    if st.checkbox("Afficher les valeurs manquantes") : 
        st.dataframe(df.isna().sum())
        


elif page == pages[1]:
    st.write("### Dataviz")
    
    fig = plt.figure()
    sns.countplot(x = 'Survived', data = df)
    st.pyplot(fig)

    
    fig = plt.figure()
    sns.countplot(x = 'Pclass', data = df)
    st.pyplot(fig)
    
    hist = px.histogram(df, x = "Survived" ,color = "Sex", barmode = "group")
    st.plotly_chart(hist)
    
    hist_bis = px.sunburst(df, path = ['Sex', 'Pclass'])
    st.plotly_chart(hist_bis)
    

elif page == pages[2]:
    
    st.write("### Modélisation")
    
    df = df.dropna()
    
    
    df = df.drop(['Name', 'Sex','Ticket', 'Cabin', 'Embarked'], axis = 1)
    
    X = df.drop('Survived', axis =1)
    y = df['Survived']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)
    
    model_choisi = st.selectbox(label = "Choix de mon modÃ¨le", options = ['Regression Log', 'Decision Tree', 'KNN'])
    
    def train_model(model_choisi) : 
        if model_choisi == 'Regression Log' :
            model = LogisticRegression()
        elif model_choisi == 'Decision Tree' :
            model = DecisionTreeClassifier()
        elif model_choisi == 'KNN' :
            model = KNeighborsClassifier()
        model.fit(X_train, y_train) # type: ignore
        score = model.score(X_test, y_test) # type: ignore
        return score
    
    score = train_model(model_choisi)
    st.write(f"Score test : {score:.2f}")
    

    code = '''def hello():
    print("Hello, Streamlit!")'''
    st.code(code, language='python')
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    