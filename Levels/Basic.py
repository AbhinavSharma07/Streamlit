import streamlit as st
import pandas as pd

st.title("Basic Data Display")
file = st.file_uploader("Upload CSV", type=["csv"])

if file:
    data = pd.read_csv(file)
    st.write(data)
