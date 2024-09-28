import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

st.title("Simple Data App")
file = st.file_uploader("Upload CSV", type=["csv"])

if file:
    data = pd.read_csv(file)
    st.write(data.head())
    
    if st.checkbox("Show Summary"):
        st.write(data.describe())
    
    if st.checkbox("Train Model"):
        target = st.selectbox("Select Target", data.columns)
        features = st.multiselect("Select Features", data.columns)
        
        if target and features:
            X = data[features]
            y = data[target]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
            
            model = LinearRegression().fit(X_train, y_train)
            st.write("Model Trained. Score:", model.score(X_test, y_test))
