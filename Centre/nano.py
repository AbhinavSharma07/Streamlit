import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Page Config
st.set_page_config(page_title="Simple Data App", layout="wide")

# Helper functions
@st.cache_data
def load_data(file):
    return pd.read_csv(file)

def train_linear_regression(X, y):
    model = LinearRegression()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    return model, mse

# Sidebar - File uploader
st.sidebar.title("Menu")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
menu_options = st.sidebar.radio("Choose Action", ["Home", "Explore Data", "Visualize", "Machine Learning"])

# Home Page
if menu_options == "Home":
    st.title("Welcome to Simple Data Analysis App")
    st.write("This app allows basic data exploration, visualization, and machine learning tasks.")
    st.write("Upload a CSV file to get started!")

# Explore Data Page
if menu_options == "Explore Data" and uploaded_file:
    data = load_data(uploaded_file)
    st.title("Data Overview")
    st.dataframe(data.head())
    st.write("Summary Statistics")
    st.write(data.describe())
    st.write("Missing Values")
    st.write(data.isnull().sum())

# Data Visualization Page
if menu_options == "Visualize" and uploaded_file:
    data = load_data(uploaded_file)
    st.title("Data Visualization")
    columns = data.columns.tolist()
    plot_type = st.selectbox("Choose Plot Type", ["Scatter", "Line", "Bar"])
    x_axis = st.selectbox("X-Axis", columns)
    y_axis = st.selectbox("Y-Axis", columns)

    if plot_type == "Scatter":
        fig = px.scatter(data, x=x_axis, y=y_axis)
    elif plot_type == "Line":
        fig = px.line(data, x=x_axis, y=y_axis)
    else:
        fig = px.bar(data, x=x_axis, y=y_axis)
    
    st.plotly_chart(fig)

# Machine Learning - Linear Regression
if menu_options == "Machine Learning" and uploaded_file:
    data = load_data(uploaded_file)
    st.title("Machine Learning: Linear Regression")
    
    target_column = st.selectbox("Select Target Variable", data.columns)
    feature_columns = st.multiselect("Select Feature Variables", data.columns)
    
    if st.button("Train Model"):
        if feature_columns and target_column:
            X = data[feature_columns].dropna()
            y = data[target_column].dropna()
            model, mse = train_linear_regression(X, y)
            st.write(f"Mean Squared Error: {mse:.2f}")
        else:
            st.error("Please select target and feature variables")

# Extra Features Placeholder
# You can add new features here
# For example, additional models, data cleaning, exporting data, etc.

st.sidebar.markdown("---")
st.sidebar.markdown("Simple Data App built with Streamlit.")
