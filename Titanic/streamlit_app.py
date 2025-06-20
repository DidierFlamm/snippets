# -*- coding: utf-8 -*-
import streamlit as st

mp3_url = "https://archive.org/download/celine-dion-my-heart-will-go-on_202207/Celine%20Dion%20-%20My%20Heart%20Will%20Go%20On.mp3"
st.sidebar.audio(mp3_url, format="audio/mp3", loop=True, autoplay=True)

pages = [
    st.Page("./pages/1_Accueil.py", title="Accueil", icon="🛳️"),
    st.Page("./pages/2_Visualisation.py", title="Visualisation", icon="📊"),
    st.Page("./pages/3_Evaluation.py", title="Evaluation", icon="📝"),
    st.Page("./pages/4_Optimisation.py", title="Optimisation", icon="📈"),
    st.Page("./pages/5_Predictions.py", title="Predictions", icon="🎯"),
]
pg = st.navigation(pages, position="top")
pg.run()
