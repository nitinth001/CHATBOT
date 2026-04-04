import streamlit as st
import os
import tempfile
import shutil
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

# Configuration
DB_PATH = "vectorstore/db_faiss"
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
llm = ChatGroq(model="llama-3.1-8b-instant", groq_api_key=os.getenv("GROQ_API_KEY"))

st.set_page_config(page_title="Local PDF Chat", layout="wide")
st.title("📄 Local PDF Chatbot (Streamlit + FAISS)")

# Sidebar
with st.sidebar:
    st.header("Upload Documents")
    uploaded_files = st.file_uploader("Choose PDFs", type="pdf", accept_multiple_files=True)
    process_btn = st.button("Index Files")
    
    if st.button("Reset Knowledge Base"):
        if os.path.exists("vectorstore"):
            shutil.rmtree("vectorstore")
            st.rerun()

# Processing Logic
if uploaded_files and process_btn:
    all_docs = []
    with st.spinner("Processing..."):
        for uploaded_file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.getvalue())
                loader = PyPDFLoader(tmp.name)
                all_docs.extend(loader.load())
            os.remove(tmp.name)
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(all_docs)
        
        # Save locally
        vectorstore = FAISS.from_documents(splits, embeddings)
        vectorstore.save_local(DB_PATH)
        st.success("Indexing complete!")

# Chat Logic
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Ask a question...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    if os.path.exists(DB_PATH):
        # Load the local brain
        vectorstore = FAISS.load_local(DB_PATH, embeddings, allow_dangerous_deserialization=True)
        
        prompt = ChatPromptTemplate.from_template("Answer using context: {context}\nQuestion: {input}")
        combine_chain = create_stuff_documents_chain(llm, prompt)
        retrieval_chain = create_retrieval_chain(vectorstore.as_retriever(), combine_chain)
        
        response = retrieval_chain.invoke({"input": user_input})
        
        st.session_state.messages.append({"role": "assistant", "content": response['answer']})
        with st.chat_message("assistant"):
            st.markdown(response['answer'])
    else:
        st.info("Please upload a PDF to begin.")