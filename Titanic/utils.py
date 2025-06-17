import pandas as pd
import streamlit as st
import joblib


@st.cache_data
def load_csv(path: str):
    return pd.read_csv(path)


@st.cache_resource
def load_model(model_path: str):
    return joblib.load(model_path)
