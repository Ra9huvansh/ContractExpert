import streamlit as st
from transformers import pipeline
import sqlite3
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from chatbot import answer_question
import random

# Set the page config
st.set_page_config(
    page_title="Legal Risk Analyzer",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Navigation options
PAGES = {
    "Home": "🏠 Home",
    "Analyze Document": "📄 Analyze Document",
    "Predictive Analytics": "📊 Predictive Analytics",
    "AI-Powered ChatBot": "🤖 AI-Powered ChatBot",
    "About": "ℹ️ About"
}

# Sidebar navigation
st.sidebar.title("Navigation")
choice = st.sidebar.radio("Go to", list(PAGES.keys()))

# Define risky clause categories and load models
categories = ["payment risk", "date issue", "liability clause", "agreement risk"]
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", framework="pt")

# Function to load cases from the database
def load_cases():
    conn = sqlite3.connect('cases.db')
    c = conn.cursor()
    c.execute("SELECT case_text, judgment_summary, tags FROM legal_cases")
    cases = c.fetchall()
    conn.close()
    return cases

# Function to find similar cases
def find_similar_cases(uploaded_text, cases, top_n=5):
    case_texts = [case[0] for case in cases]
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([uploaded_text] + case_texts)
    cosine_similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
    similar_indices = cosine_similarities.argsort()[-top_n:][::-1]
    similar_cases = [(cases[i], cosine_similarities[i], random.uniform(0, 1)) for i in similar_indices]  # Add random prediction score
    return similar_cases

# Home Page
if choice == "Home":
    st.title("🏠 Welcome to the Legal Risk Analyzer")
    st.markdown(
        """
        **Analyze legal documents for potential risks, such as payment issues, date clauses, and more.**
        - Upload a document and let AI flag potentially risky clauses.
        - Powered by Hugging Face and Streamlit.
        
        ### Features:
        - 🚀 Fast and lightweight analysis.
        - ⚖️ Designed for legal professionals.
        - 📊 Clear, structured insights.
        """
    )
    st.image("https://images.unsplash.com/photo-1575505586569-646b2ca898fc", caption="Legal Document Analysis", use_container_width=True)

# Analyze Document Page
elif choice == "Analyze Document":
    st.title("📄 Analyze Document")
    uploaded_file = st.file_uploader("Upload a legal document", type=["txt", "docx", "pdf"])

    if uploaded_file is not None:
        # Read and display the document
        document_text = uploaded_file.read().decode("utf-8")
        st.subheader("Uploaded Document:")
        st.write(document_text)

        # Split the document into sentences
        st.write("Analyzing document for risky clauses...")
        sentences = document_text.split(".")
        risky_sentences = []

        for sentence in sentences:
            if len(sentence.strip()) > 0:
                result = classifier(sentence, candidate_labels=categories)
                if result['scores'][0] > 0.8:  # Confidence threshold
                    risky_sentences.append((sentence.strip(), result['labels'][0], result['scores'][0]))

        # Display results
        st.subheader("Flagged Risky Clauses:")
        if risky_sentences:
            for clause, category, score in risky_sentences:
                st.write(f"**Clause:** {clause}")
                st.write(f"**Category:** {category} | **Confidence:** {score:.2f}")
                st.markdown("---")
        else:
            st.success("No risky clauses detected!")

# Predictive Analytics Page
elif choice == "Predictive Analytics":
    st.title("📊 Predictive Analytics")
    uploaded_file = st.file_uploader("Upload a legal document for analysis", type=["txt"])

    if uploaded_file is not None:
        uploaded_text = uploaded_file.read().decode("utf-8")
        st.write("Uploaded Case Text:")
        st.write(uploaded_text)

        st.write("Searching for similar cases...")
        cases = load_cases()  # Load cases from the database
        similar_cases = find_similar_cases(uploaded_text, cases)  # Find similar cases

        st.subheader("Similar Cases Found:")
        for i, (case, similarity, prediction_score) in enumerate(similar_cases):
            st.write(f"**Case {i+1}:**")
            st.write(f"**Similarity Score:** {similarity:.2f}")
            st.write(f"**Prediction Score:** {prediction_score:.2f}")
            st.write(f"**Judgment Summary:** {case[1]}")
            st.write(f"**Tags:** {case[2]}")
            st.markdown("---")

# AI-Powered ChatBot Page
elif choice == "AI-Powered ChatBot":
    st.title("🤖 AI-Powered ChatBot")
    uploaded_file = st.file_uploader("Upload a legal document", type=["txt", "docx", "pdf"])

    if uploaded_file is not None:
        # Read and display the document
        context = uploaded_file.read().decode("utf-8")
        st.subheader("Uploaded Document:")
        st.write(context)

        # Chat interface
        st.subheader("Ask Your Questions")
        user_question = st.text_input("Type your question here:")

        if user_question:
            answer = answer_question(user_question, context)
            st.write(f"**Answer:** {answer}")

# About Page
elif choice == "About":
    st.title("ℹ️ About This App")
    st.markdown(
        """
        **Legal Risk Analyzer** is a web application designed to help legal professionals:
        - Detect potential risks in legal documents.
        - Identify risky clauses with AI-powered analysis.
        - Summarize legal documents for easy review.

        ### Technologies Used:
        - **Streamlit** for creating the web app.
        - **Hugging Face** for zero-shot classification and summarization.
        - **Python** for development.

        #### Developer Notes:
        - Clean, modular design.
        - Easily extensible for future features.

        **Contact**: [Raghuvansh](mailto:yourname@example.com)
        """
    )
    st.image("https://via.placeholder.com/900x300?text=Powered+by+AI+and+Streamlit", use_container_width=True)

# Footer
st.sidebar.markdown("---")
st.sidebar.write("Developed by [Raghuvansh](mailto:yourname@example.com) | ⚖️ Legal Tech")
        