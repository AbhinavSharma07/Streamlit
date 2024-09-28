import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier

st.title("Multimodel Comparison")
file = st.file_uploader("Upload CSV", type=["csv"])

if file:
    data = pd.read_csv(file)
    target = st.selectbox("Select Target", data.columns)
    features = st.multiselect("Select Features", data.columns)

    if target and features:
        X = data[features]
        y = data[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        # Linear Regression
        lr_model = LinearRegression().fit(X_train, y_train)
        st.write("Linear Regression Score:", lr_model.score(X_test, y_test))
        
        # KNN
        knn_model = KNeighborsClassifier().fit(X_train, y_train)
        st.write("KNN Accuracy:", knn_model.score(X_test, y_test))
