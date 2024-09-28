import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier

# Function to load and preprocess data
def load_data(file):
    data = pd.read_csv(file)
    if st.checkbox("Drop Missing Values"):
        data = data.dropna()
    return data

# Function to plot data
def plot_data(data, x, y):
    fig = px.scatter(data, x=x, y=y)
    st.plotly_chart(fig)

# Function to train models
def train_models(X, y):
    models = {
        "Linear Regression": LinearRegression(),
        "K-Nearest Neighbors": KNeighborsClassifier()
    }
    scores = {}
    for name, model in models.items():
        model.fit(X, y)
        scores[name] = model.score(X_test, y_test)
    return scores

# Main app logic
st.title("Data Analysis App")
file = st.file_uploader("Upload CSV", type=["csv"])

if file:
    data = load_data(file)
    st.write(data.head(), "Summary:", data.describe(), "Missing Values:", data.isnull().sum())

    if st.checkbox("Show Plot"):
        x_axis = st.selectbox("X-Axis", data.columns)
        y_axis = st.selectbox("Y-Axis", data.columns)
        plot_data(data, x_axis, y_axis)

    target = st.selectbox("Select Target", data.columns)
    features = st.multiselect("Select Features", data.columns)

    if st.button("Train Models"):
        X = data[features]
        y = data[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        scores = train_models(X_train, y_train)
        for name, score in scores.items():
            st.write(f"{name} Score: {score:.2f}")
