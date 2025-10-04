"""
- Simple authentication (SQLite, hashed passwords)
- Dashboard with KPIs and simple charts
- Guard/employee CRUD (SQLite)
- Incident reporting and viewing
- Patrol scheduling
- Camera input for check-in (st.camera_input)
- Access logs and export
- Settings and sample data initialization

Run:
1. pip install -r requirements.txt
2. streamlit run streamlit_security_app.py

Requirements (example):
streamlit
pandas
matplotlib
sqlalchemy
bcrypt

"""

import streamlit as st
import pandas as pd
import sqlite3
import hashlib
import os
import datetime
import matplotlib.pyplot as plt
from io import BytesIO

DB_PATH = "security_app.db"

# ---------------------- Utilities ----------------------

def get_db_connection():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db_connection()
    c = conn.cursor()
    # users table
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            full_name TEXT,
            role TEXT DEFAULT 'guard'
        )
    ''')
    # guards/employees
    c.execute('''
        CREATE TABLE IF NOT EXISTS employees (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            phone TEXT,
            email TEXT,
            role TEXT,
            notes TEXT
        )
    ''')
    # incidents
    c.execute('''
        CREATE TABLE IF NOT EXISTS incidents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT,
            description TEXT,
            severity TEXT,
            reported_by TEXT,
            timestamp TEXT,
            image BLOB
        )
    ''')
    # patrols
    c.execute('''
        CREATE TABLE IF NOT EXISTS patrols (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            guard_id INTEGER,
            start_time TEXT,
            end_time TEXT,
            route TEXT,
            notes TEXT
        )
    ''')
    # access logs
    c.execute('''
        CREATE TABLE IF NOT EXISTS access_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user TEXT,
            action TEXT,
            location TEXT,
            timestamp TEXT
        )
    ''')
    conn.commit()
    conn.close()


def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()


def create_default_admin():
    conn = get_db_connection()
    c = conn.cursor()
    try:
        pw = hash_password("admin123")
        c.execute("INSERT OR IGNORE INTO users (username, password_hash, full_name, role) VALUES (?,?,?,?)",
                  ("admin", pw, "Administrator", "admin"))
        conn.commit()
    except Exception as e:
        print("Could not create default admin:", e)
    finally:
        conn.close()


# ---------------------- Auth ----------------------


def authenticate(username, password):
    conn = get_db_connection()
    c = conn.cursor()
    try:
        c.execute("SELECT * FROM users WHERE username = ?", (username,))
        row = c.fetchone()
        if row:
            if hash_password(password) == row["password_hash"]:
                return dict(row)
        return None
    finally:
        conn.close()


def register_user(username, password, full_name, role="guard"):
    conn = get_db_connection()
    c = conn.cursor()
    try:
        pw = hash_password(password)
        c.execute("INSERT INTO users (username, password_hash, full_name, role) VALUES (?,?,?,?)",
                  (username, pw, full_name, role))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()


# ---------------------- CRUD & Features ----------------------


def add_employee(name, phone, email, role, notes):
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("INSERT INTO employees (name, phone, email, role, notes) VALUES (?,?,?,?,?)",
              (name, phone, email, role, notes))
    conn.commit()
    conn.close()


def list_employees():
    conn = get_db_connection()
    df = pd.read_sql_query("SELECT * FROM employees", conn)
    conn.close()
    return df


def add_incident(title, description, severity, reported_by, image_bytes=None):
    conn = get_db_connection()
    c = conn.cursor()
    ts = datetime.datetime.utcnow().isoformat()
    c.execute("INSERT INTO incidents (title, description, severity, reported_by, timestamp, image) VALUES (?,?,?,?,?,?)",
              (title, description, severity, reported_by, ts, image_bytes))
    conn.commit()
    conn.close()


def list_incidents():
    conn = get_db_connection()
    df = pd.read_sql_query("SELECT id, title, severity, reported_by, timestamp FROM incidents ORDER BY timestamp DESC", conn)
    conn.close()
    return df


def get_incident_image(incident_id):
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("SELECT image FROM incidents WHERE id = ?", (incident_id,))
    row = c.fetchone()
    conn.close()
    if row and row[0]:
        return row[0]
    return None


def add_patrol(guard_id, start_time, end_time, route, notes):
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("INSERT INTO patrols (guard_id, start_time, end_time, route, notes) VALUES (?,?,?,?,?)",
              (guard_id, start_time, end_time, route, notes))
    conn.commit()
    conn.close()


def list_patrols():
    conn = get_db_connection()
    df = pd.read_sql_query("SELECT * FROM patrols ORDER BY start_time DESC", conn)
    conn.close()
    return df


def log_access(user, action, location):
    conn = get_db_connection()
    c = conn.cursor()
    ts = datetime.datetime.utcnow().isoformat()
    c.execute("INSERT INTO access_logs (user, action, location, timestamp) VALUES (?,?,?,?)",
              (user, action, location, ts))
    conn.commit()
    conn.close()


def list_access_logs(limit=200):
    conn = get_db_connection()
    df = pd.read_sql_query("SELECT * FROM access_logs ORDER BY timestamp DESC LIMIT {}".format(limit), conn)
    conn.close()
    return df


# ---------------------- Streamlit App ----------------------


def main():
    st.set_page_config(page_title="Security Company App", layout="wide")

    if not os.path.exists(DB_PATH):
        init_db()
        create_default_admin()

    if "user" not in st.session_state:
        st.session_state.user = None

    # --- Sidebar: Auth & Navigation ---
    st.sidebar.title("Security App")

    if st.session_state.user is None:
        auth_choice = st.sidebar.selectbox("Auth", ["Login", "Register"])
        if auth_choice == "Login":
            username = st.sidebar.text_input("Username")
            password = st.sidebar.text_input("Password", type="password")
            if st.sidebar.button("Login"):
                user = authenticate(username, password)
                if user:
                    st.session_state.user = user
                    st.success(f"Welcome {user.get('full_name') or user.get('username')}!")
                else:
                    st.error("Invalid credentials")
        else:
            new_user = st.sidebar.text_input("New username")
            new_full = st.sidebar.text_input("Full name")
            new_pwd = st.sidebar.text_input("Password", type="password")
            role = st.sidebar.selectbox("Role", ["guard", "supervisor", "admin"]) 
            if st.sidebar.button("Register"):
                ok = register_user(new_user, new_pwd, new_full, role)
                if ok:
                    st.success("User registered. Please login from sidebar.")
                else:
                    st.error("Username already exists.")
    else:
        st.sidebar.markdown(f"**Logged in as:** {st.session_state.user.get('full_name') or st.session_state.user.get('username')} (<{st.session_state.user.get('role')}>)")
        if st.sidebar.button("Logout"):
            st.session_state.user = None
            st.experimental_rerun()

    # --- Main Navigation ---
    pages = ["Dashboard", "Incidents", "Employees", "Patrols", "Access Logs", "Settings"]
    if st.session_state.user:
        if st.session_state.user.get('role') != 'admin':
            pages.remove('Settings')
    page = st.sidebar.radio("Go to", pages)

    # --- Dashboard ---
    if page == "Dashboard":
        st.title("Security Operations Dashboard")
        col1, col2, col3 = st.columns(3)
        incidents_df = list_incidents()
        logs_df = list_access_logs(100)
        employees_df = list_employees()
        col1.metric("Incidents (total)", len(incidents_df))
        col2.metric("Access log events", len(logs_df))
        col3.metric("Employees", len(employees_df))

        st.subheader("Incidents by severity")
        if not incidents_df.empty:
            sev = incidents_df['severity'].value_counts()
            fig, ax = plt.subplots()
            ax.pie(sev, labels=sev.index, autopct='%1.1f%%')
            ax.axis('equal')
            st.pyplot(fig)
        else:
            st.info("No incidents reported yet.")

        st.subheader("Recent Access Logs")
        st.dataframe(logs_df.head(20))

    # --- Incidents ---
    if page == "Incidents":
        st.title("Incidents")
        st.subheader("Report an incident")
        with st.form("incident_form"):
            t = st.text_input("Title")
            desc = st.text_area("Description")
            sev = st.selectbox("Severity", ["Low", "Medium", "High", "Critical"] )
            img = st.camera_input("Take a picture (optional)")
            submitted = st.form_submit_button("Report")
            if submitted:
                image_bytes = None
                if img is not None:
                    image_bytes = img.getvalue()
                reporter = st.session_state.user.get('username') if st.session_state.user else 'anonymous'
                add_incident(t, desc, sev, reporter, image_bytes)
                st.success("Incident reported")
                log_access(reporter, f"reported_incident:{t}", "N/A")

        st.subheader("All incidents")
        inc_df = list_incidents()
        st.dataframe(inc_df)

        sel = st.number_input("Open incident ID (to view details)", min_value=0, step=1)
        if sel > 0:
            conn = get_db_connection()
            c = conn.cursor()
            c.execute("SELECT * FROM incidents WHERE id = ?", (sel,))
            row = c.fetchone()
            conn.close()
            if row:
                st.markdown(f"### {row['title']} | Severity: {row['severity']}")
                st.write(row['description'])
                st.write(f"Reported by: {row['reported_by']} at {row['timestamp']}")
                if row['image']:
                    st.image(row['image'])
            else:
                st.info("No incident with that ID")

    # --- Employees ---
    if page == "Employees":
        st.title("Employees / Guards")
        st.subheader("Add employee")
        with st.form("emp_form"):
            name = st.text_input("Name")
            phone = st.text_input("Phone")
            email = st.text_input("Email")
            role = st.selectbox("Role", ["guard", "supervisor", "manager"])
            notes = st.text_area("Notes")
            sub = st.form_submit_button("Add")
            if sub:
                add_employee(name, phone, email, role, notes)
                st.success("Employee added")

        st.subheader("All employees")
        edf = list_employees()
        st.dataframe(edf)

    # --- Patrols ---
    if page == "Patrols":
        st.title("Patrol Schedules")
        st.subheader("Create patrol")
        emp_df = list_employees()
        emp_options = emp_df.set_index('id')['name'].to_dict() if not emp_df.empty else {}
        with st.form("patrol_form"):
            guard = st.selectbox("Guard", options=list(emp_options.keys()) if emp_options else [], format_func=lambda x: emp_options.get(x, "(none)"))
            start = st.datetime_input("Start time", value=datetime.datetime.now())
            end = st.datetime_input("End time", value=datetime.datetime.now() + datetime.timedelta(hours=8))
            route = st.text_input("Route / Areas to cover")
            notes = st.text_area("Notes")
            subp = st.form_submit_button("Schedule")
            if subp:
                add_patrol(guard, start.isoformat(), end.isoformat(), route, notes)
                st.success("Patrol scheduled")
                log_access(st.session_state.user.get('username') if st.session_state.user else 'system', "create_patrol", route)

        st.subheader("Recent patrols")
        pdf = list_patrols()
        st.dataframe(pdf)

    # --- Access Logs ---
    if page == "Access Logs":
        st.title("Access Logs")
        logs = list_access_logs(500)
        st.dataframe(logs)
        if st.button("Export logs CSV"):
            csv = logs.to_csv(index=False).encode()
            st.download_button("Download CSV", data=csv, file_name="access_logs.csv", mime="text/csv")

    # --- Settings (admin only) ---
    if page == "Settings":
        st.title("Settings & Maintenance")
        st.subheader("Initialize sample data")
        if st.button("Create sample employees + incidents"):
            add_employee('John Doe', '9999999999', 'john@example.com', 'guard', 'Night shift')
            add_employee('Jane Smith', '8888888888', 'jane@example.com', 'supervisor', 'Day shift')
            add_incident('Suspicious Activity', 'Person loitering at gate', 'Medium', 'admin')
            st.success("Sample data created")

        st.subheader("Manage Admins")
        st.write("Current users")
        conn = get_db_connection()
        users_df = pd.read_sql_query('SELECT id, username, full_name, role FROM users', conn)
        conn.close()
        st.dataframe(users_df)

        st.subheader("Purge all data (Danger)")
        if st.button("Purge DB - DANGEROUS"):
            confirm = st.text_input("Type 'CONFIRM' to purge")
            if confirm == 'CONFIRM':
                os.remove(DB_PATH)
                st.warning("Database deleted. App will reinitialize on refresh.")


if __name__ == '__main__':
    main()
