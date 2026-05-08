import streamlit as st
from PyPDF2 import PdfReader
from dotenv import load_dotenv
import os

from langchain_groq import ChatGroq

load_dotenv()

# Page Config
st.set_page_config(
    page_title="AI Resume Analyzer",
    page_icon="📄",
    layout="centered"
)

st.title("📄 AI Resume Analyzer")
st.write("Upload your resume and get AI feedback")

# Upload PDF
uploaded_file = st.file_uploader(
    "Upload Resume (PDF)",
    type="pdf"
)

# Read Skills
with open("skills.txt", "r") as file:
    skills_db = file.read().splitlines()

if uploaded_file:

    # Read PDF
    pdf_reader = PdfReader(uploaded_file)

    resume_text = ""

    for page in pdf_reader.pages:
        resume_text += page.extract_text()

    st.subheader("Resume Content")
    st.text_area("", resume_text, height=200)

    # Skill Detection
    detected_skills = []

    for skill in skills_db:
        if skill.lower() in resume_text.lower():
            detected_skills.append(skill)

    st.subheader("✅ Detected Skills")
    st.write(detected_skills)

    # ATS Score
    ats_score = min(len(detected_skills) * 5, 100)

    st.subheader("📊 ATS Score")
    st.progress(ats_score)
    st.write(f"ATS Score: {ats_score}/100")

    # AI Feedback
    llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama-3.1-8b-instant"
)

    prompt = f"""
    Analyze this resume and provide:

    1. Strengths
    2. Weaknesses
    3. Missing skills
    4. Resume improvement suggestions
    5. Recommended job roles

    Resume:
    {resume_text}
    """

    response = llm.invoke(prompt)

    st.subheader("🤖 AI Feedback")
    st.write(response.content)