import streamlit as st
from dotenv import load_dotenv
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

# Load Environment Variables
load_dotenv()

# Streamlit Page
st.set_page_config(page_title="PDF RAG Chatbot")
st.title("📄 PDF RAG Chatbot")
st.write("Upload a PDF and ask questions from the document")

# Upload PDF
uploaded_file = st.file_uploader(
    "Upload PDF File",
    type="pdf"
)

if uploaded_file is not None:

    # Save Uploaded PDF
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    st.success("PDF Uploaded Successfully")

    # Load PDF
    loader = PyPDFLoader("temp.pdf")
    docs = loader.load()

    # Split Text
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    final_documents = text_splitter.split_documents(docs)

    # Ollama Embeddings
    embeddings = OllamaEmbeddings(
        model="nomic-embed-text"
    )

    # Vector Store
    vectorstore = Chroma.from_documents(
        documents=final_documents,
        embedding=embeddings
    )

    # Retriever
    retriever = vectorstore.as_retriever()

    # Groq LLM
    llm = ChatGroq(
        groq_api_key=os.environ["GROQ_API_KEY"],
        model_name="llama3-8b-8192"
    )

    # Prompt
    prompt = ChatPromptTemplate.from_template("""
    You are an intelligent AI assistant specialized in answering questions from uploaded PDF documents.

    Instructions:
    - Use ONLY the provided PDF context to answer the question.
    - Give accurate and concise answers.
    - If the answer is not found in the context, say:
      "The answer is not available in the uploaded PDF document."

    PDF Context:
    {context}

    User Question:
    {input}

    Answer:
    """)

    # Stuff Document Chain
    document_chain = create_stuff_documents_chain(
        llm,
        prompt
    )

    # Retrieval Chain
    retrieval_chain = create_retrieval_chain(
        retriever,
        document_chain
    )

    # User Input
    user_question = st.text_input(
        "Ask a Question from the PDF"
    )

    if user_question:

        response = retrieval_chain.invoke({
            "input": user_question
        })

        st.subheader("Answer")
        st.write(response["answer"])