import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

st.title("Simple Data App")

# Upload CSV file
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.write("Data Preview", data.head())

    # Sidebar for navigation
    option = st.sidebar.selectbox("Choose Action", ["Explore Data", "Visualize", "Train Model"])

    # Data exploration
    if option == "Explore Data":
        st.write("Summary Statistics", data.describe())
        st.write("Missing Values", data.isnull().sum())

    # Data visualization
    elif option == "Visualize":
        columns = data.columns
        x_axis = st.selectbox("Select X-Axis", columns)
        y_axis = st.selectbox("Select Y-Axis", columns)
        plot_type = st.radio("Select Plot Type", ["Scatter", "Line", "Bar"])
        
        if plot_type == "Scatter":
            fig = px.scatter(data, x=x_axis, y=y_axis)
        elif plot_type == "Line":
            fig = px.line(data, x=x_axis, y=y_axis)
        else:
            fig = px.bar(data, x=x_axis, y=y_axis)
        
        st.plotly_chart(fig)

    # Train a Linear Regression model
    elif option == "Train Model":
        target = st.selectbox("Select Target", data.columns)
        features = st.multiselect("Select Features", data.columns)

        if features and target:
            X = data[features]
            y = data[target]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            model = LinearRegression()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)

            st.write(f"Mean Squared Error: {mse:.2f}")
