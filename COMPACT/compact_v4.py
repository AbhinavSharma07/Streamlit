import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Title and description
st.title("Comprehensive Streamlit App")
st.write("This app demonstrates various Streamlit features, including input widgets, layouts, and data visualization.")

# Sidebar input
st.sidebar.title("Sidebar Controls")
age = st.sidebar.slider("Select your age:", 1, 100, 25)
country = st.sidebar.selectbox("Select your country:", ["USA", "Canada", "UK", "Germany", "India"])

# Layout management with columns
col1, col2 = st.columns(2)

with col1:
    st.subheader("User Information")
    name = st.text_input("Enter your name:")
    if st.button("Greet"):
        st.write(f"Hello, {name}! You are {age} years old and live in {country}.")

with col2:
    st.subheader("Simple Calculator")
    number1 = st.number_input("Enter the first number:")
    number2 = st.number_input("Enter the second number:")
    operation = st.selectbox("Select operation:", ["Add", "Subtract", "Multiply", "Divide"])
    
    if st.button("Calculate"):
        if operation == "Add":
            result = number1 + number2
        elif operation == "Subtract":
            result = number1 - number2
        elif operation == "Multiply":
            result = number1 * number2
        else:
            result = number1 / number2 if number2 != 0 else "Division by zero error"
        st.write(f"Result: {result}")

# Data visualization
st.subheader("Random Data Visualization")
data = pd.DataFrame(np.random.randn(100, 3), columns=["A", "B", "C"])
st.line_chart(data)

# Interactive Matplotlib Plot
st.subheader("Interactive Matplotlib Plot")

x = np.linspace(0, 10, 100)
y = np.sin(x) * age  # Using age from the sidebar to make the plot interactive

fig, ax = plt.subplots()
ax.plot(x, y)        
ax.set_title(f"Sine Wave Modulated by Age ({age})")
st.pyplot(fig)

# Display data table
st.subheader("Random Data Table")
st.write(data.head())

# Footer
st.write("Thanks for exploring this app!") 
