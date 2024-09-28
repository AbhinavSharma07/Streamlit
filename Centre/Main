import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import plotly.express as px
from io import BytesIO

# Set page configuration
st.set_page_config(
    page_title="Comprehensive Data Analysis App",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Function to load data
@st.cache_data
def load_data(file):
    try:
        data = pd.read_csv(file)
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None
    return data

# Function to download data
def download_data(data, filename="processed_data.csv"):
    towrite = BytesIO()
    data.to_csv(towrite, index=False)
    towrite.seek(0)
    return towrite

# Sidebar Navigation
st.sidebar.title("Navigation")
options = st.sidebar.radio("Go to", [
    "Home",
    "Upload Data",
    "Data Exploration",
    "Data Visualization",
    "Machine Learning",
    "Download Data",
    "About"
])

# Home Page
if options == "Home":
    st.title("Welcome to the Comprehensive Data Analysis App")
    st.write("""
    This application allows you to upload your datasets, explore them, visualize various aspects,
    perform machine learning tasks, and download the processed data. Use the sidebar to navigate
    through different sections of the app.
    """)
    st.image("https://images.unsplash.com/photo-1555066931-4365d14bab8c?ixlib=rb-4.0.3&auto=format&fit=crop&w=1350&q=80", use_column_width=True)

# Upload Data Page
elif options == "Upload Data":
    st.title("Upload Your Dataset")
    st.write("""
    Upload a CSV file to get started. The app will display the data and provide options for
    further analysis.
    """)
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded_file is not None:
        data = load_data(uploaded_file)
        if data is not None:
            st.success("Data successfully loaded!")
            st.subheader("Data Preview")
            st.dataframe(data.head())
            # Store data in session state
            st.session_state['data'] = data
    else:
        st.info("Awaiting CSV file to be uploaded.")

# Data Exploration Page
elif options == "Data Exploration":
    st.title("Data Exploration")
    if 'data' not in st.session_state:
        st.warning("Please upload data in the 'Upload Data' section first.")
    else:
        data = st.session_state['data']
        st.subheader("Dataset Overview")
        st.write(f"**Number of Rows:** {data.shape[0]}")
        st.write(f"**Number of Columns:** {data.shape[1]}")

        st.subheader("Column Names and Data Types")
        st.dataframe(pd.DataFrame({
            "Column": data.columns,
            "Data Type": data.dtypes.values
        }))

        st.subheader("Missing Values")
        missing_values = data.isnull().sum()
        st.dataframe(pd.DataFrame({
            "Column": data.columns,
            "Missing Values": missing_values
        }))

        st.subheader("Summary Statistics")
        st.write(data.describe())

# Data Visualization Page
elif options == "Data Visualization":
    st.title("Data Visualization")
    if 'data' not in st.session_state:
        st.warning("Please upload data in the 'Upload Data' section first.")
    else:
        data = st.session_state['data']
        st.sidebar.subheader("Visualization Settings")

        # Select visualization type
        vis_type = st.sidebar.selectbox("Select Visualization Type", [
            "Line Chart",
            "Bar Chart",
            "Scatter Plot",
            "Histogram",
            "Box Plot",
            "Correlation Heatmap"
        ])

        if vis_type == "Line Chart":
            st.subheader("Line Chart")
            columns = data.select_dtypes(include=np.number).columns.tolist()
            x_axis = st.sidebar.selectbox("X-axis", columns)
            y_axis = st.sidebar.selectbox("Y-axis", columns)
            if x_axis and y_axis:
                fig, ax = plt.subplots()
                ax.plot(data[x_axis], data[y_axis], marker='o')
                ax.set_xlabel(x_axis)
                ax.set_ylabel(y_axis)
                ax.set_title(f"{y_axis} vs {x_axis}")
                st.pyplot(fig)

        elif vis_type == "Bar Chart":
            st.subheader("Bar Chart")
            columns = data.select_dtypes(include=['object', 'category']).columns.tolist()
            if not columns:
                st.warning("No categorical columns available for bar chart.")
            else:
                category = st.sidebar.selectbox("Category", columns)
                count = data[category].value_counts().reset_index()
                count.columns = [category, "Count"]
                fig = px.bar(count, x=category, y="Count", title=f"Count of {category}")
                st.plotly_chart(fig, use_container_width=True)

        elif vis_type == "Scatter Plot":
            st.subheader("Scatter Plot")
            numeric_cols = data.select_dtypes(include=np.number).columns.tolist()
            if len(numeric_cols) < 2:
                st.warning("Need at least two numeric columns for scatter plot.")
            else:
                x_axis = st.sidebar.selectbox("X-axis", numeric_cols, key="scatter_x")
                y_axis = st.sidebar.selectbox("Y-axis", numeric_cols, key="scatter_y")
                color = st.sidebar.selectbox("Color By", [None] + data.columns.tolist(), key="scatter_color")
                fig = px.scatter(data, x=x_axis, y=y_axis, color=color, title=f"Scatter Plot of {y_axis} vs {x_axis}")
                st.plotly_chart(fig, use_container_width=True)

        elif vis_type == "Histogram":
            st.subheader("Histogram")
            numeric_cols = data.select_dtypes(include=np.number).columns.tolist()
            if not numeric_cols:
                st.warning("No numeric columns available for histogram.")
            else:
                column = st.sidebar.selectbox("Select Column", numeric_cols)
                bins = st.sidebar.slider("Number of Bins", min_value=5, max_value=100, value=30)
                fig = px.histogram(data, x=column, nbins=bins, title=f"Histogram of {column}")
                st.plotly_chart(fig, use_container_width=True)

        elif vis_type == "Box Plot":
            st.subheader("Box Plot")
            numeric_cols = data.select_dtypes(include=np.number).columns.tolist()
            if not numeric_cols:
                st.warning("No numeric columns available for box plot.")
            else:
                column = st.sidebar.selectbox("Select Column", numeric_cols)
                fig = px.box(data, y=column, title=f"Box Plot of {column}")
                st.plotly_chart(fig, use_container_width=True)

        elif vis_type == "Correlation Heatmap":
            st.subheader("Correlation Heatmap")
            numeric_data = data.select_dtypes(include=np.number)
            if numeric_data.empty:
                st.warning("No numeric data available for correlation heatmap.")
            else:
                corr = numeric_data.corr()
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
                st.pyplot(fig)

# Machine Learning Page
elif options == "Machine Learning":
    st.title("Machine Learning: Linear Regression")
    if 'data' not in st.session_state:
        st.warning("Please upload data in the 'Upload Data' section first.")
    else:
        data = st.session_state['data']
        st.subheader("Prepare Data for Training")

        # Select target variable
        target = st.selectbox("Select Target Variable", data.columns.tolist())

        # Select features
        features = st.multiselect("Select Feature Variables", [col for col in data.columns if col != target])

        if st.button("Train Model"):
            if not features:
                st.error("Please select at least one feature.")
            else:
                # Handle missing values
                dataset = data[[target] + features].dropna()
                X = dataset[features]
                y = dataset[target]

                # Encode categorical variables
                X = pd.get_dummies(X, drop_first=True)

                # Split data
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )

                # Train model
                model = LinearRegression()
                model.fit(X_train, y_train)

                # Predict
                y_pred = model.predict(X_test)

                # Evaluate
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                r2 = r2_score(y_test, y_pred)

                st.success("Model Trained Successfully!")
                st.subheader("Model Performance")
                st.write(f"**Mean Squared Error (MSE):** {mse:.2f}")
                st.write(f"**Root Mean Squared Error (RMSE):** {rmse:.2f}")
                st.write(f"**R-squared (R²):** {r2:.2f}")

                # Plot Actual vs Predicted
                fig, ax = plt.subplots()
                ax.scatter(y_test, y_pred, alpha=0.7)
                ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
                ax.set_xlabel("Actual")
                ax.set_ylabel("Predicted")
                ax.set_title("Actual vs Predicted")
                st.pyplot(fig)

# Download Data Page
elif options == "Download Data":
    st.title("Download Processed Data")
    if 'data' not in st.session_state:
        st.warning("Please upload and process data in previous sections first.")
    else:
        data = st.session_state['data']
        st.write("You can download the current state of your data below.")
        csv = data.to_csv(index=False)
        st.download_button(
            label="Download data as CSV",
            data=csv,
            file_name='processed_data.csv',
            mime='text/csv',
        )

# About Page
elif options == "About":
    st.title("About This App")
    st.write("""
    **Comprehensive Data Analysis App** is built using Streamlit to provide an interactive platform for
    data scientists, analysts, and enthusiasts to upload, explore, visualize, and perform basic
    machine learning tasks on their datasets.

    **Features:**
    - **Data Upload:** Easily upload CSV files.
    - **Data Exploration:** View summaries, data types, and missing values.
    - **Data Visualization:** Create various interactive plots.
    - **Machine Learning:** Train a simple Linear Regression model and evaluate its performance.
    - **Download Data:** Download the processed data for further use.

    **Developed By:** [Your Name]

    **License:** MIT

    **Contact:** your.email@example.com
    """)
    st.image("https://images.unsplash.com/photo-1529333166437-7750a6dd5a70?ixlib=rb-4.0.3&auto=format&fit=crop&w=1350&q=80", use_column_width=True)

# Footer
st.markdown("""
---
**Comprehensive Data Analysis App** built with ❤️ using [Streamlit](https://streamlit.io/)
""")
