import streamlit as st
from PyPDF2 import PdfReader
from dotenv import load_dotenv
import os


from langchain_groq import ChatGroq
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings


load_dotenv()


st.set_page_config(
    page_title="AI Resume Analyzer",
    page_icon="📄",
    layout="centered"
)


st.markdown("""
<style>
.stButton button {
    background-color: #ff4b4b;
    color: white;
    border-radius: 10px;
}
.stProgress > div > div {
    background-color: green;
}
</style>
""", unsafe_allow_html=True)


st.title("📄 AI Resume Analyzer")
st.write("Upload your resume and get AI-powered analysis")

uploaded_file = st.file_uploader(
    "Upload Resume (PDF)",
    type="pdf"
)

with open("skills.txt", "r") as file:
    skills_db = file.read().splitlines()

if uploaded_file:

    pdf_reader = PdfReader(uploaded_file)

    resume_text = ""

    for page in pdf_reader.pages:
        text = page.extract_text()

        if text:
            resume_text += text

    st.subheader("📄 Resume Content")

    st.text_area(
        "Extracted Text",
        resume_text,
        height=250
    )

    

    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=500,
        chunk_overlap=50
    )

    texts = splitter.split_text(resume_text)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_texts(
        texts,
        embeddings
    )

    docs = vectorstore.similarity_search(
        "Analyze this resume and evaluate technical skills"
    )

    retrieved_text = "\n".join(
        [doc.page_content for doc in docs]
    )

  

    detected_skills = []

    for skill in skills_db:

        if skill.lower() in resume_text.lower():
            detected_skills.append(skill)

    st.subheader("✅ Detected Skills")

    if detected_skills:
        st.write(detected_skills)
    else:
        st.write("No matching skills detected.")

   

    ats_score = min(len(detected_skills) * 5, 100)

    st.subheader("📊 ATS Score")

    st.progress(ats_score)

    st.write(f"ATS Score: {ats_score}/100")

  

    try:

        # Initialize Groq LLM
        llm = ChatGroq(
            api_key=os.getenv("GROQ_API_KEY"),
            model="llama-3.1-8b-instant"
        )

        prompt = f"""
        Analyze this resume and provide:

        1. Strengths
        2. Weaknesses
        3. Missing skills
        4. Resume improvement suggestions
        5. Recommended job roles
        6. Overall evaluation

        Resume Content:
        {retrieved_text}
        """

        
        response = llm.invoke([
            ("human", prompt)
        ])

      
        st.subheader("🤖 AI Feedback")

        st.write(response.content)

    except Exception as e:

        st.error("Error generating AI feedback.")

        st.error(e)
