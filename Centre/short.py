# save this as app.py

import streamlit as st

st.title("👋 Welcome to My Streamlit App")

# Text input from the user
name = st.text_input("What's your name?")

# Button to submit
if st.button("Greet Me"):
    if name:
        st.success(f"Hello, {name}! 👋")
    else:
        st.warning("Please enter your name first.")
