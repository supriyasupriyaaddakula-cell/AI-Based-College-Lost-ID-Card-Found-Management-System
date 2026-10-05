import streamlit as st
import sqlite3
import pandas as pd
from sentence_transformers import SentenceTransformer
from datetime import datetime
import re


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Campus ID Recovery",
    page_icon="🪪",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# PROFESSIONAL CSS
# ============================================================

st.markdown("""
<style>

.stApp {
    background:
        radial-gradient(circle at top left, #172554 0%, transparent 35%),
        radial-gradient(circle at bottom right, #312e81 0%, transparent 35%),
        linear-gradient(135deg, #020617, #0f172a);
    color: white;
}

section[data-testid="stSidebar"] {
    background: #020617;
    border-right: 1px solid #1e293b;
}

.hero {
    padding: 35px;
    border-radius: 22px;
    background: linear-gradient(
        135deg,
        rgba(37,99,235,0.25),
        rgba(124,58,237,0.20)
    );
    border: 1px solid rgba(255,255,255,0.10);
    margin-bottom: 30px;
}

.hero h1 {
    font-size: 42px;
    font-weight: 800;
    margin-bottom: 8px;
}

.hero p {
    color: #cbd5e1;
    font-size: 17px;
}

.stat {
    background: rgba(255,255,255,0.07);
    border: 1px solid rgba(255,255,255,0.10);
    border-radius: 18px;
    padding: 22px;
    text-align: center;
}

.stat-number {
    font-size: 32px;
    font-weight: 800;
    color: #60a5fa;
}

.stat-label {
    color: #cbd5e1;
    font-size: 14px;
}

.card {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.10);
    border-radius: 18px;
    padding: 24px;
    margin-bottom: 20px;
}

.feature {
    background: rgba(255,255,255,0.05);
    padding: 16px;
    border-radius: 12px;
    margin: 8px 0;
}

.match-card {
    background: rgba(16,185,129,0.10);
    border: 1px solid rgba(16,185,129,0.35);
    border-radius: 16px;
    padding: 20px;
    margin: 12px 0;
}

.warning-card {
    background: rgba(245,158,11,0.10);
    border: 1px solid rgba(245,158,11,0.30);
    border-radius: 16px;
    padding: 18px;
}

.footer {
    text-align: center;
    color: #64748b;
    padding: 35px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# AI MODEL
# ============================================================

@st.cache_resource
def load_ai_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


with st.spinner("🧠 Loading AI matching system..."):
    model = load_ai_model()


# ============================================================
# DATABASE
# ============================================================

DB_NAME = "lost_found.db"


def get_connection():
    return sqlite3.connect(DB_NAME)


def create_database():

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            report_type TEXT NOT NULL,
            student_name TEXT NOT NULL,
            roll_number TEXT NOT NULL,
            branch TEXT,
            year TEXT,
            card_number TEXT,
            location TEXT,
            description TEXT,
            contact TEXT,
            date_reported TEXT,
            status TEXT DEFAULT 'Active'
        )
    """)

    conn.commit()
    conn.close()


create_database()


# ============================================================
# DATABASE FUNCTIONS
# ============================================================

def add_report(
    report_type,
    student_name,
    roll_number,
    branch,
    year,
    card_number,
    location,
    description,
    contact
):

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO reports
        (
            report_type,
            student_name,
            roll_number,
            branch,
            year,
            card_number,
            location,
            description,
            contact,
            date_reported,
            status
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        report_type,
        student_name,
        roll_number,
        branch,
        year,
        card_number,
        location,
        description,
        contact,
        datetime.now().strftime("%Y-%m-%d %H:%M"),
        "Active"
    ))

    conn.commit()
    conn.close()


def get_reports():

    conn = get_connection()

    df = pd.read_sql_query(
        "SELECT * FROM reports ORDER BY id DESC",
        conn
    )

    conn.close()

    return df


def update_status(report_id, status):

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        "UPDATE reports SET status=? WHERE id=?",
        (status, report_id)
    )

    conn.commit()
    conn.close()


def delete_report(report_id):

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        "DELETE FROM reports WHERE id=?",
        (report_id,)
    )

    conn.commit()
    conn.close()


# ============================================================
# AI MATCHING
# ============================================================

def clean_text(text):

    if text is None:
        return ""

    return str(text).lower().strip()


def create_matching_text(row):

    return f"""
    roll number {clean_text(row['roll_number'])}
    card number {clean_text(row['card_number'])}
    branch {clean_text(row['branch'])}
    year {clean_text(row['year'])}
    location {clean_text(row['location'])}
    description {clean_text(row['description'])}
    """


def find_matches(lost_record, found_records):

    if found_records.empty:
        return []

    lost_text = create_matching_text(lost_record)

    lost_embedding = model.encode(
        lost_text,
        normalize_embeddings=True
    )

    results = []

    for _, found in found_records.iterrows():

        found_text = create_matching_text(found)

        found_embedding = model.encode(
            found_text,
            normalize_embeddings=True
        )

        score = float(
            lost_embedding @ found_embedding
        )

        percentage = max(
            0,
            min(100, score * 100)
        )

        results.append({
            "id": found["id"],
            "student_name": found["student_name"],
            "roll_number": found["roll_number"],
            "branch": found["branch"],
            "location": found["location"],
            "description": found["description"],
            "score": percentage,
            "status": found["status"]
        })

    results.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    return results


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.markdown("## 🪪 Campus ID Recovery")

st.sidebar.caption(
    "AI-Based Lost & Found Management"
)

st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Dashboard",
        "🚨 Report Lost ID",
        "🔎 Report Found ID",
        "🤖 AI Match",
        "📋 All Reports"
    ]
)

st.sidebar.markdown("---")

st.sidebar.markdown("### ⚙️ Technology")

st.sidebar.write("🐍 Python")
st.sidebar.write("🎨 Streamlit")
st.sidebar.write("🧠 Sentence Transformers")
st.sidebar.write("🗄️ SQLite")
st.sidebar.write("📊 Pandas")

st.sidebar.markdown("---")

st.sidebar.success("System Online 🟢")


# ============================================================
# HEADER
# ============================================================

st.markdown("""
<div class="hero">

<h1>🪪 Campus ID Recovery</h1>

<p>
AI-Based College Lost ID Card & Found Management System
</p>

</div>
""", unsafe_allow_html=True)


# ============================================================
# DASHBOARD
# ============================================================

if page == "🏠 Dashboard":

    df = get_reports()

    total = len(df)

    lost_count = len(
        df[df["report_type"] == "Lost"]
    ) if not df.empty else 0

    found_count = len(
        df[df["report_type"] == "Found"]
    ) if not df.empty else 0

    resolved_count = len(
        df[df["status"] == "Returned"]
    ) if not df.empty else 0

    # Statistics

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown(f"""
        <div class="stat">
            <div class="stat-number">{total}</div>
            <div class="stat-label">Total Reports</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="stat">
            <div class="stat-number">{lost_count}</div>
            <div class="stat-label">Lost Cards</div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div class="stat">
            <div class="stat-number">{found_count}</div>
            <div class="stat-label">Found Cards</div>
        </div>
        """, unsafe_allow_html=True)

    with c4:
        st.markdown(f"""
        <div class="stat">
            <div class="stat-number">{resolved_count}</div>
            <div class="stat-label">Returned</div>
        </div>
        """, unsafe_allow_html=True)

    st.write("")

    # Project information

    st.markdown("""
    <div class="card">

    <h2>🌟 About the Project</h2>

    <p>
    Campus ID Recovery is an AI-powered college management system
    designed to help students report lost ID cards and allow college
    administrators to manage found cards efficiently.
    </p>

    <h3>🚀 Professional Features</h3>

    <div class="feature">
    🚨 <b>Lost ID Reporting</b><br>
    Students can report missing college ID cards.
    </div>

    <div class="feature">
    🔎 <b>Found ID Reporting</b><br>
    Students or staff can register an ID card that they found.
    </div>

    <div class="feature">
    🤖 <b>AI Matching</b><br>
    AI compares lost and found card information and generates
    a similarity score.
    </div>

    <div class="feature">
    📊 <b>Management Dashboard</b><br>
    View total lost, found and returned cards.
    </div>

    <div class="feature">
    📥 <b>Report Export</b><br>
    Download all records as CSV for administration.
    </div>

    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 🔄 How the System Works")

    a, b, c = st.columns(3)

    with a:
        st.info(
            "1️⃣ Student reports a lost ID card"
        )

    with b:
        st.info(
            "2️⃣ Someone reports a found ID card"
        )

    with c:
        st.success(
            "3️⃣ AI finds the best possible match"
        )


# ============================================================
# REPORT LOST ID
# ============================================================

elif page == "🚨 Report Lost ID":

    st.markdown("## 🚨 Report Lost ID Card")

    st.warning(
        "Please provide accurate information so the AI can find possible matches."
    )

    with st.form("lost_form"):

        c1, c2 = st.columns(2)

        with c1:

            name = st.text_input(
                "Student Name *"
            )

            roll = st.text_input(
                "Roll Number *"
            )

            branch = st.selectbox(
                "Branch",
                [
                    "CSE",
                    "CSD / Data Science",
                    "AI & ML",
                    "IT",
                    "ECE",
                    "EEE",
                    "Other"
                ]
            )

            year = st.selectbox(
                "Year",
                [
                    "1st Year",
                    "2nd Year",
                    "3rd Year",
                    "4th Year"
                ]
            )

        with c2:

            card_number = st.text_input(
                "ID Card Number"
            )

            location = st.text_input(
                "Last Seen Location",
                placeholder="Example: Library"
            )

            description = st.text_area(
                "Description",
                placeholder="Describe where and how you lost the ID card..."
            )

            contact = st.text_input(
                "Contact Number / Email *"
            )

        submit = st.form_submit_button(
            "🚨 Submit Lost ID Report",
            use_container_width=True
        )

    if submit:

        if not name or not roll or not contact:

            st.error(
                "Please fill all required fields."
            )

        else:

            add_report(
                "Lost",
                name,
                roll,
                branch,
                year,
                card_number,
                location,
                description,
                contact
            )

            st.success(
                "✅ Lost ID card report submitted successfully!"
            )

            st.info(
                "🤖 You can now use AI Match to find possible found cards."
            )


# ============================================================
# REPORT FOUND ID
# ============================================================

elif page == "🔎 Report Found ID":

    st.markdown("## 🔎 Report Found ID Card")

    st