import streamlit as st
import pandas as pd

st.title("Interactive Data Filter")
file = st.file_uploader("Upload CSV", type=["csv"])

if file:
    data = pd.read_csv(file)
    column = st.selectbox("Select Column to Filter", data.columns)
    unique_values = data[column].unique()
    selected_values = st.multiselect("Select Values to Filter", unique_values, unique_values)

    filtered_data = data[data[column].isin(selected_values)]
    st.write(filtered_data)
