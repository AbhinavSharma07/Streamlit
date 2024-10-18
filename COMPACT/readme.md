# Data Analysis App 📊

This **Streamlit** application allows users to upload a CSV file, visualize data, and train simple machine learning models like **Linear Regression** and **K-Nearest Neighbors (KNN)**.

## Features 🚀
- **CSV Upload:** Upload any dataset in CSV format.  
- **Data Overview:** Display the first few rows, summary statistics, and missing value counts.  
- **Data Cleaning:** Drop rows with missing values if required.  
- **Interactive Plotting:** Visualize data using interactive scatter plots with Plotly.  
- **Model Training:** Train and evaluate Linear Regression and KNN models using selected features.


## Install dependencies:

streamlit
pandas
plotly
scikit-learn


## Example Code :

'''
import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier

st.title("Data Analysis App")
file = st.file_uploader("Upload CSV", type=["csv"])

if file:
    data = pd.read_csv(file)
    st.write(data.head(), "Summary:", data.describe(), "Missing Values:", data.isnull().sum())

    if st.checkbox("Drop Missing Values"):
        data = data.dropna()

    # Data Visualization
    if st.checkbox("Show Plot"):
        x_axis = st.selectbox("X-Axis", data.columns)
        y_axis = st.selectbox("Y-Axis", data.columns)
        st.plotly_chart(px.scatter(data, x=x_axis, y=y_axis))

    # Train Models
    target = st.selectbox("Select Target", data.columns)
    features = st.multiselect("Select Features", data.columns)

    if st.button("Train Models"):
        X = data[features]
        y = data[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        # Linear Regression
        lr_model = LinearRegression().fit(X_train, y_train)
        lr_score = lr_model.score(X_test, y_test)

        # KNN
        knn_model = KNeighborsClassifier().fit(X_train, y_train)
        knn_score = knn_model.score(X_test, y_test)

        st.write("Linear Regression Score:", lr_score)
        st.write("KNN Accuracy:", knn_score)
'''
