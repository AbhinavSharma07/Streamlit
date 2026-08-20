import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import os

# --- Utility / helper functions ---

@st.cache_data
def load_sample_data(n=100):
    """Generate some random sample data with latitude / longitude."""
    df = pd.DataFrame({
        "lat": np.random.randn(n) / 50 + 37.76,
        "lon": np.random.randn(n) / 50 - 122.4,
        "value": np.random.randint(1, 100, size=n),
        "time": [datetime.datetime.now() - datetime.timedelta(days=int(x)) for x in np.random.randint(0, 30, size=n)]
    })
    return df

def plot_hobby_time(hobbies):
    """Return a bar chart figure for given hobbies."""
    hours = [np.random.randint(1, 10) for _ in hobbies]
    fig, ax = plt.subplots()
    ax.bar(hobbies, hours, color="skyblue")
    ax.set_ylabel("Hours (estimated)")
    ax.set_title("Estimated Time Spent (Weekly)")
    return fig

# --- Main app structure ---

def main():
    st.set_page_config(page_title="Full Featured App", layout="wide", page_icon="🚀")

    # Navigation (simple multipage pattern)
    pages = {
        "Home": page_home,
        "Profile": page_profile,
        "Map & Data": page_map_data,
        "Chat / Notes": page_chat,
    }
    st.sidebar.title("Navigation")
    choice = st.sidebar.radio("Go to", list(pages.keys()))

    # Run the selected page
    pages[choice]()

    # Footer
    st.sidebar.markdown("---")
    st.sidebar.write("Built with 🧡 using Streamlit")

# --- Page definitions ---

def page_home():
    st.title("Welcome to the Full Featured Streamlit App")
    st.write(
        """
        This app shows many features in one place:

        - Multi‑page navigation  
        - Sidebar & layout  
        - File upload (image)  
        - Charts, maps & data  
        - Using caching, session state  
        - Simple chat / notes  
        """
    )
    st.markdown("## Sample Data Preview")
    df = load_sample_data(20)
    st.dataframe(df.head())

    st.markdown("## Sample Line Chart")
    df2 = pd.DataFrame(np.random.randn(30, 3), columns=["A", "B", "C"])
    st.line_chart(df2)

    st.markdown("## Trigger Some Action")
    if st.button("Show Balloons"):
        st.balloons()

def page_profile():
    st.header("🧑 User Profile")

    # Use session state to persist data across pages
    if "profile_uploaded" not in st.session_state:
        st.session_state.profile_uploaded = False

    col1, col2 = st.columns([1, 2])

    with col1:
        uploaded_file = st.file_uploader("Upload Profile Picture (jpeg/png)", type=["jpg", "jpeg", "png"])
        if uploaded_file:
            st.image(uploaded_file, caption="Your profile picture", use_column_width=True)
            st.session_state.profile_uploaded = True
        else:
            if not st.session_state.profile_uploaded:
                st.info("No profile image uploaded yet.")

    with col2:
        with st.form("profile_form", clear_on_submit=False):
            name = st.text_input("Name")
            age = st.number_input("Age", min_value=0, max_value=120, step=1)
            bio = st.text_area("Bio / About You")
            hobbies = st.multiselect("Select your hobbies", ["Art", "Gaming", "Reading", "Running", "Music", "Yoga"])
            mood = st.selectbox("How are you feeling today?", ["😊 Happy", "😐 Neutral", "😞 Sad"])
            submitted = st.form_submit_button("Save Profile")

            if submitted:
                st.success("Profile saved!")
                # Store into session state for persistence
                st.session_state["profile"] = {
                    "name": name,
                    "age": age,
                    "bio": bio,
                    "hobbies": hobbies,
                    "mood": mood,
                }

    # Display summary if profile exists
    if "profile" in st.session_state:
        p = st.session_state["profile"]
        st.subheader("Your Profile Summary")
        st.write(f"**Name:** {p.get('name', '')}")
        st.write(f"**Age:** {p.get('age', '')}")
        st.write(f"**Mood:** {p.get('mood', '')}")
        st.write("**Bio:**")
        st.write(p.get("bio", ""))
        if p.get("hobbies"):
            fig = plot_hobby_time(p["hobbies"])
            st.pyplot(fig)

def page_map_data():
    st.header("🌍 Map & Data Viewer")

    df = load_sample_data(200)
    st.subheader("Raw Data")
    st.write(df.head())

    # Show map
    st.subheader("Map View")
    st.map(df[["lat", "lon"]])

    # Filtering by value
    st.subheader("Filter by Value")
    min_val = int(df["value"].min())
    max_val = int(df["value"].max())
    filt = st.slider("Value Range", min_val, max_val, (min_val, max_val))
    filtered = df[(df["value"] >= filt[0]) & (df["value"] <= filt[1])]
    st.write(f"Showing {len(filtered)} points")
    st.map(filtered[["lat", "lon"]])

def page_chat():
    st.header("💬 Chat / Notes")

    if "notes" not in st.session_state:
        st.session_state.notes = []

    with st.form("note_form", clear_on_submit=True):
        name = st.text_input("Your name", value="Anonymous")
        message = st.text_area("Message / Note")
        send = st.form_submit_button("Post Message")
        if send and message.strip():
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            st.session_state.notes.append(f"**{name}** at {timestamp}:  {message}")

    st.subheader("Messages")
    for msg in reversed(st.session_state.notes[-20:]):
        st.markdown(msg)

# Entry point
if __name__ == "__main__":
    main()
