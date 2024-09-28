import streamlit as st
import pandas as pd

st.title("Data Preprocessing")
file = st.file_uploader("Upload CSV", type=["csv"])

if file:
    data = pd.read_csv(file)
    if st.checkbox("Drop Missing Values"):
        data = data.dropna()
    st.write(data.head())
    st.write("Missing Values after cleaning:", data.isnull().sum())
