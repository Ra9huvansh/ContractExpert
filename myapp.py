import streamlit as st

# -------------------- Page Configuration --------------------
st.set_page_config(
    page_title="Legal Risk Analyzer",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

from transformers import pipeline
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import sqlite3
import streamlit_authenticator as stauth
import PyPDF2
import docx

# Clear session state on startup
if "logged_in" not in st.session_state:
    st.session_state.clear()  # Clear session on app start
    st.session_state.logged_in = False  # Ensure the user is not logged in

# -------------------- Define Risk Categories --------------------
categories = [
    "Payment Risk",
    "Confidentiality Breach",
    "Termination Clause",
    "Dispute Resolution Issue",
    "Date Clauses",
    "Force Majeure",
]

# -------------------- Database Initialization --------------------
def init_user_db():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

init_user_db()

# -------------------- Authentication Setup --------------------
def get_users():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute("SELECT username, email, password FROM users")
    rows = c.fetchall()
    conn.close()
    return {
        row[0]: {
            "name": row[0],
            "email": row[1],
            "password": row[2],
        }
        for row in rows
    }

users = get_users()

authenticator = stauth.Authenticate(
    {"usernames": users},
    "unique_cookie_name",
    "very_secure_random_key",
    cookie_expiry_days=0,
)

# -------------------- Load Models --------------------
@st.cache_resource
def load_classifier():
    return pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

@st.cache_resource
def load_sentence_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

classifier = load_classifier()
sentence_model = load_sentence_model()

# -------------------- Helper Functions --------------------
def load_cases():
    conn = sqlite3.connect('cases.db')
    c = conn.cursor()
    c.execute("SELECT case_text, judgment_summary, tags FROM legal_cases")
    cases = c.fetchall()
    conn.close()
    return cases

def find_similar_cases(uploaded_text, cases, top_n=5):
    case_texts = [case[0] for case in cases]
    embeddings = sentence_model.encode([uploaded_text] + case_texts)
    cosine_similarities = cosine_similarity([embeddings[0]], embeddings[1:]).flatten()
    similar_indices = cosine_similarities.argsort()[-top_n:][::-1]
    similar_cases = [(cases[i], cosine_similarities[i]) for i in similar_indices]
    return similar_cases

# -------------------- Sidebar Navigation --------------------
st.sidebar.title("Navigation")
auth_status = authenticator.login(location='sidebar')

if not auth_status:
    PAGES = {"Login": "🔑 Login", "Register": "📝 Register"}
else:
    PAGES = {
        "Home": "🏠 Home",
        "Analyze Document": "📄 Analyze Document",
        "Predictive Analytics": "📊 Predictive Analytics",
        "About": "ℹ️ About",
    }

choice = st.sidebar.radio("Go to", list(PAGES.keys()))

# -------------------- Registration Page --------------------
if choice == "Register":
    st.title("📝 Register")
    username = st.text_input("Username")
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")
    confirm_password = st.text_input("Confirm Password", type="password")

    if st.button("Register"):
        if password != confirm_password:
            st.error("Passwords do not match!")
        elif not username or not email or not password:
            st.error("Please fill out all fields.")
        else:
            try:
                conn = sqlite3.connect('users.db')
                c = conn.cursor()
                c.execute("INSERT INTO users (username, email, password) VALUES (?, ?, ?)", 
                          (username, email, password))
                conn.commit()
                conn.close()
                st.success("Registration successful! You can now log in.")
            except sqlite3.IntegrityError:
                st.error("Username or email already exists.")
            except Exception as e:
                st.error(f"An error occurred: {e}")

# -------------------- Authenticated Pages --------------------
if auth_status and choice in PAGES:
    if choice == "Home":
        st.title("🏠 Welcome to the Legal Risk Analyzer")
        st.markdown("""
        **Analyze legal documents for potential risks.**
        - Upload a document and let AI flag risky clauses.
        """)
    
    elif choice == "Analyze Document":
        st.title("📄 Analyze Document")
        uploaded_file = st.file_uploader("Upload a legal document", type=["txt", "docx", "pdf"])

        if uploaded_file:
            try:
                if uploaded_file.type == "application/pdf":
                    reader = PyPDF2.PdfReader(uploaded_file)
                    document_text = "".join([page.extract_text() for page in reader.pages])
                elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                    doc = docx.Document(uploaded_file)
                    document_text = "\n".join([para.text for para in doc.paragraphs])
                else:
                    document_text = uploaded_file.read().decode("utf-8")

                st.subheader("Uploaded Document:")
                st.text(document_text)

                st.write("Analyzing document for risky clauses...")
                sentences = document_text.split(".")
                risky_sentences = []

                for sentence in sentences:
                    sentence = sentence.strip()
                    if sentence:
                        result = classifier(sentence, candidate_labels=categories)
                        if result['scores'][0] > 0.8:
                            risky_sentences.append((sentence, result['labels'][0], result['scores'][0]))

                st.subheader("Flagged Risky Clauses:")
                if risky_sentences:
                    for clause, category, score in risky_sentences:
                        st.write(f"**Clause:** {clause}")
                        st.write(f"**Category:** {category} | **Confidence:** {score:.2f}")
                        st.markdown("---")
                else:
                    st.success("No risky clauses detected!")
            except Exception as e:
                st.error(f"Error processing the file: {e}")

    elif choice == "Predictive Analytics":
        st.title("📊 Predictive Analytics")
        uploaded_file = st.file_uploader("Upload a legal document for analysis", type=["txt"])

        if uploaded_file:
            try:
                uploaded_text = uploaded_file.read().decode("utf-8")
                st.write("**Uploaded Case Text:**")
                st.text(uploaded_text)

                st.write("Searching for similar cases...")
                cases = load_cases()
                similar_cases = find_similar_cases(uploaded_text, cases)

                st.subheader("Similar Cases Found:")
                for i, (case, similarity) in enumerate(similar_cases):
                    st.write(f"**Case {i+1}:**")
                    st.write(f"**Similarity Score:** {similarity:.2f}")
                    st.write(f"**Judgment Summary:** {case[1]}")
                    st.write(f"**Tags:** {case[2]}")
                    st.markdown("---")
            except Exception as e:
                st.error(f"Error during analysis: {e}")

    elif choice == "About":
        st.title("ℹ️ About This App")
        st.markdown(
            """
            **Legal Risk Analyzer** is a web application designed to help legal professionals:
            - Detect potential risks in legal documents.
            - Identify risky clauses with AI-powered analysis.
    
            ### Technologies Used:
            - **Streamlit** for creating the web app.
            - **Hugging Face** for zero-shot classification.
            - **Python** for development.
    
            #### Developer Notes:
            - Clean, modular design.
            - Easily extensible for future features.
    
            **Contact**: [Your Name](mailto:yourname@example.com)
            """
        )
        st.image(
            "https://via.placeholder.com/900x300?text=Powered+by+AI+and+Streamlit", 
            use_container_width=True
        )
    
    # -------------------- Footer --------------------
    st.sidebar.markdown("---")
    authenticator.logout("Logout", "sidebar")
    st.sidebar.write("Developed by [Your Name](mailto:yourname@example.com) | ⚖️ Legal Tech")
