import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from io import StringIO

# ------------------------------
# Caching: Expensive computations are cached to improve performance
@st.cache_data
def load_data():
    # Simulate loading data
    data = pd.DataFrame({
        'A': np.random.randn(100),
        'B': np.random.randn(100),
        'C': np.random.randn(100),
        'D': np.random.randint(0, 100, 100)
    })
    return data

# ------------------------------
# Session State: Manage state across interactions
if 'count' not in st.session_state:
    st.session_state.count = 0

# ------------------------------
# Sidebar: Navigation and Controls
st.sidebar.title("Navigation")
app_mode = st.sidebar.radio("Go to", ["Home", "Data Input", "Data Visualization", "Settings"])

st.sidebar.header("Controls")
theme = st.sidebar.selectbox("Select Theme", ["Light", "Dark", "Colorful"])
st.sidebar.write(f"**Selected Theme:** {theme}")

# Apply theme (Basic demonstration)
if theme == "Light":
    st.markdown(
        """
        <style>
        body {
            background-color: white;
            color: black;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
elif theme == "Dark":
    st.markdown(
        """
        <style>
        body {
            background-color: #2E2E2E;
            color: white;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
elif theme == "Colorful":
    st.markdown(
        """
        <style>
        body {
            background: linear-gradient(135deg, #f6d365 0%, #fda085 100%);
            color: black;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

# ------------------------------
# Home Page
if app_mode == "Home":
    st.title("📈 Comprehensive Streamlit App")
    st.write("""
        Welcome to the comprehensive Streamlit application! This app demonstrates various features of Streamlit, including:
        - **User Inputs**: Text, numbers, sliders, selectboxes, checkboxes, radio buttons, and file uploads.
        - **Data Handling**: Displaying and manipulating dataframes.
        - **Data Visualization**: Using Matplotlib, Seaborn, and Plotly for interactive plots.
        - **Layout Management**: Using columns, tabs, and expanders.
        - **State Management**: Managing session state.
        - **Caching**: Improving performance with cached functions.
        - **File Downloads**: Allowing users to download processed data.
    """)

# ------------------------------
# Data Input Page
elif app_mode == "Data Input":
    st.title("📥 Data Input and Manipulation")
    
    st.header("User Inputs")
    
    # Text Input
    name = st.text_input("Enter your name:", "John Doe")
    
    # Number Input
    age = st.number_input("Enter your age:", min_value=0, max_value=120, value=30)
    
    # Slider
    salary = st.slider("Select your expected salary (in $1000s):", 30, 200, 50)
    
    # Selectbox
    country = st.selectbox("Select your country:", ["USA", "Canada", "UK", "Germany", "India"])
    
    # Multi-select
    skills = st.multiselect("Select your skills:", ["Python", "Java", "C++", "JavaScript", "SQL"])
    
    # Radio Buttons
    experience = st.radio("Do you have prior work experience?", ["Yes", "No"])
    
    # Checkbox
    willing_to_relocate = st.checkbox("Willing to relocate")
    
    # Display Inputs
    st.subheader("Your Information")
    st.write(f"**Name:** {name}")
    st.write(f"**Age:** {age}")
    st.write(f"**Expected Salary:** ${salary * 1000}")
    st.write(f"**Country:** {country}")
    st.write(f"**Skills:** {', '.join(skills) if skills else 'None'}")
    st.write(f"**Prior Experience:** {experience}")
    st.write(f"**Willing to Relocate:** {'Yes' if willing_to_relocate else 'No'}")
    
    # File Upload
    st.header("📂 Upload Your CSV File")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)
            st.success("File uploaded successfully!")
            st.subheader("Data Preview")
            st.dataframe(data.head())
        except Exception as e:
            st.error(f"Error: {e}")
    
    # Download Button
    st.header("💾 Download Your Information as CSV")
    user_info = pd.DataFrame({
        "Name": [name],
        "Age": [age],
        "Expected Salary": [salary * 1000],
        "Country": [country],
        "Skills": [", ".join(skills) if skills else "None"],
        "Prior Experience": [experience],
        "Willing to Relocate": ["Yes" if willing_to_relocate else "No"]
    })
    
    csv = user_info.to_csv(index=False)
    st.download_button(
        label="Download Data as CSV",
        data=csv,
        file_name='user_info.csv',
        mime='text/csv',
    )

# ------------------------------
# Data Visualization Page
elif app_mode == "Data Visualization":
    st.title("📊 Data Visualization")
    
    st.header("🔄 Refresh Data")
    if st.button("Load Data"):
        data = load_data()
        st.success("Data loaded successfully!")
    else:
        data = load_data()
    
    st.subheader("Raw Data")
    st.write(data.head())
    
    # Matplotlib Plot
    st.header("📉 Matplotlib Line Chart")
    fig, ax = plt.subplots()
    ax.plot(data['A'], label='A')
    ax.plot(data['B'], label='B')
    ax.set_xlabel('Index')
    ax.set_ylabel('Value')
    ax.set_title('Line Chart using Matplotlib')
    ax.legend()
    st.pyplot(fig)
    
    # Seaborn Heatmap
    st.header("🔥 Seaborn Heatmap")
    corr = data.corr()
    fig, ax = plt.subplots()
    sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)
    
    # Plotly Interactive Scatter Plot
    st.header("✨ Plotly Interactive Scatter Plot")
    fig = px.scatter(data, x='A', y='B', color='C', size='D',
                     title="Scatter Plot using Plotly",
                     labels={"A": "Feature A", "B": "Feature B"})
    st.plotly_chart(fig)
    
    # Interactive Map (Using sample data)
    st.header("🗺️ Interactive Map")
    map_data = pd.DataFrame(
        np.random.randn(1000, 2) / [50, 50] + [37.76, -122.4],
        columns=['lat', 'lon']
    )
    st.map(map_data)
    
    # Data Filtering
    st.header("🔍 Filter Data")
    min_a, max_a = st.slider("Select range for A:", float(data['A'].min()), float(data['A'].max()), (float(data['A'].min()), float(data['A'].max())))
    filtered_data = data[(data['A'] >= min_a) & (data['A'] <= max_a)]
    st.write(f"Filtered Data: {filtered_data.shape[0]} rows")
    st.dataframe(filtered_data)

# ------------------------------
# Settings Page
elif app_mode == "Settings":
    st.title("⚙️ Settings and Advanced Features")
    
    st.header("🛠️ Session State Management")
    st.write(f"Button clicked {st.session_state.count} times.")
    if st.button("Increment Count"):
        st.session_state.count += 1
    
    st.header("📁 File Handling")
    st.write("Upload a text file and see its content.")
    text_file = st.file_uploader("Upload a TXT file", type=["txt"])
    if text_file is not None:
        stringio = StringIO(text_file.getvalue().decode("utf-8"))
        string_data = stringio.read()
        st.text(string_data)
    
    st.header("📈 Cached Function Example")
    @st.cache_data
    def expensive_computation(x):
        import time
        time.sleep(5)  # Simulate expensive computation
        return x * x
    
    num = st.number_input("Enter a number to compute its square (cached):", value=2)
    if st.button("Compute Square"):
        result = expensive_computation(num)
        st.write(f"The square of {num} is {result}")
        st.write("Computed using cached function. Subsequent computations with the same input are faster.")
    
    st.header("🔒 Hide Streamlit Style Elements")
    st.write("You can hide certain Streamlit default elements using custom CSS.")
    hide_streamlit_style = """
                <style>
                #MainMenu {visibility: hidden;}
                footer {visibility: hidden;}
                </style>
                """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# ------------------------------
# Footer
st.markdown("""
---
© 2024 Comprehensive Streamlit App. Built with ❤️ by ABHINAV SHARMA (OKAY)
""")
