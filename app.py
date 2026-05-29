
from langchain_text_splitters import RecursiveCharacterTextSplitter
import streamlit as st
from dotenv import load_dotenv
import os
import tempfile

from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_groq import ChatGroq

# Load environment variables
load_dotenv()

# Streamlit page setup
st.set_page_config(page_title="PDF Chat App")
st.header("📄 Chat with PDF using Groq")

# API Key
groq_api_key = os.getenv("GROQ_API_KEY")

# Upload PDF
uploaded_file = st.file_uploader("Upload PDF File", type="pdf")

# User Question
user_question = st.text_input("Ask a Question from PDF")

if uploaded_file is not None:

    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        pdf_path = tmp_file.name

    # Read PDF
    pdf_reader = PdfReader(pdf_path)

    text = ""

    for page in pdf_reader.pages:
        text += page.extract_text()

    # Split Text
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    docs = text_splitter.split_text(text)

    # Embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Vector Store
    vectorstore = FAISS.from_texts(docs, embeddings)

    if user_question:

        # Similarity Search
        relevant_docs = vectorstore.similarity_search(user_question, k=3)

        context = "\n".join([doc.page_content for doc in relevant_docs])

        # Prompt
        prompt = f"""
        Answer the question based on the provided PDF context.

        Context:
        {context}

        Question:
        {user_question}
        """

        # LLM
        llm = ChatGroq(
            groq_api_key=groq_api_key,
            model_name="llama-3.1-8b-instant"
        )

        # Generate Response
        response = llm.invoke(prompt)

        st.subheader("Answer")
        st.write(response.content)