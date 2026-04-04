import streamlit as st
import os
import tempfile
import shutil
from io import BytesIO
from dotenv import load_dotenv
from reportlab.pdfgen import canvas

# LangChain Imports
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader # Added Web Loader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

# --- v3.0 STYLING ---
st.set_page_config(page_title="AI Multi-Source Explorer", layout="wide")
st.markdown("""
    <style>
    .main { background-color: #0e1117; color: #ffffff; }
    .stButton>button { border-radius: 20px; border: 1px solid #00f2fe; background-color: #0e1117; color: white; }
    .stSidebar { background-color: #161b22; }
    </style>
    """, unsafe_allow_html=True)

# Configuration
DB_PATH = "vectorstore/db_faiss"
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
llm = ChatGroq(model="llama-3.1-8b-instant", groq_api_key=os.getenv("GROQ_API_KEY"))

# --- PDF EXPORT FUNCTION ---
def generate_pdf_summary(text):
    buffer = BytesIO()
    p = canvas.Canvas(buffer)
    p.drawString(100, 800, "AI Analysis Report")
    p.line(100, 790, 500, 790)
    y = 770
    for line in text.split('\n'):
        if y < 50: p.showPage(); y = 800
        p.drawString(100, y, line[:90])
        y -= 15
    p.save(); buffer.seek(0)
    return buffer

st.title("🌐 AI Multi-Source Explorer v3.0")

# --- Sidebar: Hybrid Input ---
with st.sidebar:
    st.header("🛠️ Data Sources")
    source_type = st.radio("Choose Source:", ["📄 PDF Documents", "🔗 Website URL"])
    
    if source_type == "📄 PDF Documents":
        uploaded_files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
        process_btn = st.button("Index PDFs")
    else:
        web_url = st.text_input("Enter Website URL (e.g., https://example.com)")
        process_btn = st.button("Index Website")

    st.divider()
    if st.button("🗑️ Reset All"):
        if os.path.exists("vectorstore"): shutil.rmtree("vectorstore")
        st.session_state.messages = []
        st.rerun()

# --- v3.0 Logic: Handling PDF or Web ---
if process_btn:
    all_docs = []
    with st.spinner("Indexing Source..."):
        try:
            if source_type == "📄 PDF Documents" and uploaded_files:
                for uploaded_file in uploaded_files:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(uploaded_file.getvalue())
                        tmp_path = tmp.name
                    loader = PyPDFLoader(tmp_path)
                    all_docs.extend(loader.load())
                    os.remove(tmp_path)
            
            elif source_type == "🔗 Website URL" and web_url:
                loader = WebBaseLoader(web_url)
                all_docs.extend(loader.load())
            
            if all_docs:
                splits = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(all_docs)
                vectorstore = FAISS.from_documents(splits, embeddings)
                vectorstore.save_local(DB_PATH)
                st.success("Successfully Indexed!")
            else:
                st.warning("No data found to index.")
        except Exception as e:
            st.error(f"Error: {str(e)}")

# --- Chat Interface ---
if "messages" not in st.session_state: st.session_state.messages = []
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]): st.markdown(msg["content"])

user_input = st.chat_input("Ask about your data...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"): st.markdown(user_input)

    if os.path.exists(DB_PATH):
        vectorstore = FAISS.load_local(DB_PATH, embeddings, allow_dangerous_deserialization=True)
        prompt = ChatPromptTemplate.from_template("Answer using context: {context}\nQuestion: {input}")
        retrieval_chain = create_retrieval_chain(vectorstore.as_retriever(), create_stuff_documents_chain(llm, prompt))
        
        response = retrieval_chain.invoke({"input": user_input})
        full_response = response['answer']

        st.session_state.messages.append({"role": "assistant", "content": full_response})
        with st.chat_message("assistant"):
            st.markdown(full_response)
            st.download_button("📥 Export PDF", generate_pdf_summary(full_response), "report.pdf", "application/pdf")