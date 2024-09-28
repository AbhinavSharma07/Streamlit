import streamlit as st
import pandas as pd
import plotly.express as px

st.title("Advanced Plotting")
file = st.file_uploader("Upload CSV", type=["csv"])

if file:
    data = pd.read_csv(file)
    plot_type = st.selectbox("Choose Plot Type", ["Scatter", "Line", "Bar"])
    x_axis = st.selectbox("X-Axis", data.columns)
    y_axis = st.selectbox("Y-Axis", data.columns)

    if plot_type == "Scatter":
        fig = px.scatter(data, x=x_axis, y=y_axis)
    elif plot_type == "Line":
        fig = px.line(data, x=x_axis, y=y_axis)
    else:
        fig = px.bar(data, x=x_axis, y=y_axis)

    st.plotly_chart(fig)
