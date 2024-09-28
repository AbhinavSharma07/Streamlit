import streamlit as st
import pandas as pd

st.title("Data Summary and Missing Values")
file = st.file_uploader("Upload CSV", type=["csv"])

if file:
    data = pd.read_csv(file)
    st.write(data.head())
    st.write("Summary", data.describe())
    st.write("Missing Values", data.isnull().sum())
