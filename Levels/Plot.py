import streamlit as st
import pandas as pd
import plotly.express as px

st.title("Data Plot")
file = st.file_uploader("Upload CSV", type=["csv"])

if file:
    data = pd.read_csv(file)
    st.write(data.head())
    
    if st.checkbox("Show Scatter Plot"):
        x_axis = st.selectbox("X-Axis", data.columns)
        y_axis = st.selectbox("Y-Axis", data.columns)
        fig = px.scatter(data, x=x_axis, y=y_axis)
        st.plotly_chart(fig)
