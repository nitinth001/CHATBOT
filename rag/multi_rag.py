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
# --- Logic: Processing Multiple PDFs ---
if uploaded_files and process_btn:
    all_docs = []
    with st.spinner("Processing your PDFs..."):
        for uploaded_file in uploaded_files:
            # FIX: Write to temp file and CLOSE it before loading
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name  # Save the path
            import streamlit as st
import os
import tempfile
import shutil
from io import BytesIO
from dotenv import load_dotenv
from reportlab.pdfgen import canvas

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

# --- v2.0 CUSTOM STYLING ---
st.set_page_config(page_title="AI PDF Pro v2.0", layout="wide")
st.markdown("""
    <style>
    .main { background-color: #0e1117; color: #ffffff; }
    .stButton>button { border-radius: 20px; border: 1px solid #ff4b4b; background-color: #0e1117; color: white; }
    .stSidebar { background-color: #161b22; }
    </style>
    """, unsafe_allow_html=True)

# Configuration
DB_PATH = "vectorstore/db_faiss"
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
llm = ChatGroq(model="llama-3.1-8b-instant")

# --- PDF EXPORT FUNCTION ---
def generate_pdf_summary(text):
    buffer = BytesIO()
    p = canvas.Canvas(buffer)
    p.drawString(100, 800, "AI Generated Summary Export")
    p.line(100, 790, 500, 790)
    
    # Simple text wrapping for the PDF
    y = 770
    for line in text.split('\n'):
        if y < 50: # New page if bottom reached
            p.showPage()
            y = 800
        p.drawString(100, y, line[:90])
        y -= 15
    p.save()
    buffer.seek(0)
    return buffer

st.title("🚀 AI PDF Pro v2.0")

with st.sidebar:
    st.header("📂 Document Lab")
    uploaded_files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
    if st.button("🚀 Index Everything"):
        all_docs = []
        with st.spinner("Analyzing..."):
            for uploaded_file in uploaded_files:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_file.getvalue())
                    tmp_path = tmp.name
                loader = PyPDFLoader(tmp_path)
                all_docs.extend(loader.load())
                os.remove(tmp_path)
            
            splits = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(all_docs)
            vectorstore = FAISS.from_documents(splits, embeddings)
            vectorstore.save_local(DB_PATH)
            st.success("Indexing Complete!")

    if st.button("🗑️ Reset"):
        if os.path.exists("vectorstore"):
            shutil.rmtree("vectorstore")
            st.rerun()

# --- Chat Interface ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Analyze these documents...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    if os.path.exists(DB_PATH):
        vectorstore = FAISS.load_local(DB_PATH, embeddings, allow_dangerous_deserialization=True)
        
        # v2.0 PROMPT (Asking for Page Citations)
        prompt = ChatPromptTemplate.from_template("""
        Answer accurately using the context. 
        At the end of your answer, list the source page numbers found in metadata.
        Context: {context}
        Question: {input}""")
        
        combine_chain = create_stuff_documents_chain(llm, prompt)
        retrieval_chain = create_retrieval_chain(vectorstore.as_retriever(), combine_chain)
        
        response = retrieval_chain.invoke({"input": user_input})
        answer = response['answer']
        
        # v2.0 CITATION LOGIC: Extracting page numbers from source docs
        pages = set([str(doc.metadata.get('page', 0) + 1) for doc in response['context']])
        citation_text = f"\n\n**Sources:** Page(s) {', '.join(pages)}"
        full_response = answer + citation_text

        st.session_state.messages.append({"role": "assistant", "content": full_response})
        with st.chat_message("assistant"):
            st.markdown(full_response)
            
            # v2.0 DOWNLOAD BUTTON
            pdf_file = generate_pdf_summary(full_response)
            st.download_button(
                label="📥 Download this Summary as PDF",
                data=pdf_file,
                file_name="ai_summary.pdf",
                mime="application/pdf"
            )
    else:
        st.info("Upload a document to activate the AI.")
            # Now that the file is closed, the data is safely on disk
            try:
                loader = PyPDFLoader(tmp_path)
                all_docs.extend(loader.load())
            finally:
                # Always delete the temp file after loading
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
        
        # ... rest of your splitting and indexing code ...
        
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