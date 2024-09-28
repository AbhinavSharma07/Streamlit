import streamlit as st
import pandas as pd

st.title("Data Upload and Download")
file = st.file_uploader("Upload CSV", type=["csv"])

if file:
    data = pd.read_csv(file)
    st.write(data.head())

    csv = data.to_csv(index=False).encode('utf-8')
    st.download_button("Download Data as CSV", csv, "data.csv", "text/csv")
