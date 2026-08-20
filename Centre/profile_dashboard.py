import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import random

# Page setup
st.set_page_config(page_title="User Profile Dashboard", layout="wide", page_icon="🧑‍💻")

# Title and intro
st.title("🧑‍💻 User Profile Dashboard")
st.markdown("Fill out your details and view your profile summary below!")

# Sidebar: User Input
st.sidebar.header("📋 Enter Your Info")

# User Inputs
name = st.sidebar.text_input("Name")
age = st.sidebar.number_input("Age", min_value=0, max_value=120, step=1)
mood = st.sidebar.selectbox("How are you feeling today?", ["😊 Happy", "😐 Neutral", "😞 Sad"])
hobbies = st.sidebar.multiselect("Choose your hobbies:", 
                                 ["🎨 Art", "🎮 Gaming", "📚 Reading", "🏃‍♂️ Running", "🎵 Music", "🧘 Yoga"])

uploaded_image = st.sidebar.file_uploader("Upload a profile picture", type=["jpg", "jpeg", "png"])

# Main section layout
col1, col2 = st.columns([1, 2])

# Left column: Image and mood
with col1:
    st.subheader("🖼️ Profile Picture")
    if uploaded_image:
        st.image(uploaded_image, use_column_width=True)
    else:
        st.info("No image uploaded.")

    st.subheader("😎 Mood")
    st.write(f"You're feeling: **{mood}**")

# Right column: User info
with col2:
    st.subheader("📄 Profile Summary")
    if name and age:
        st.success(f"Welcome, **{name}**!")
        st.write(f"You're **{age}** years old.")
    else:
        st.warning("Please enter your name and age in the sidebar.")

    if hobbies:
        st.write("Your hobbies:")
        for hobby in hobbies:
            st.write(f"- {hobby}")
    else:
        st.info("No hobbies selected.")

# Divider
st.markdown("---")

# 📊 Hobby Time Chart
st.subheader("⏱️ Estimated Time Spent on Hobbies This Week")

if hobbies:
    # Generate random hours for hobbies
    data = {
        "Hobby": hobbies,
        "Hours": [random.randint(1, 10) for _ in hobbies]
    }
    df = pd.DataFrame(data)

    fig, ax = plt.subplots()
    ax.bar(df["Hobby"], df["Hours"], color='skyblue')
    ax.set_ylabel("Hours")
    ax.set_title("Time Spent on Hobbies")
    st.pyplot(fig)
else:
    st.info("Select hobbies to see a chart.")

# Footer
st.markdown("---")
st.caption("Made with ❤️ using Streamlit")
