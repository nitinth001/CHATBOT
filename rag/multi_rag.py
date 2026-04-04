import streamlit as st
import os
import tempfile
import shutil
import time
from io import BytesIO
from dotenv import load_dotenv
from reportlab.pdfgen import canvas

# --- CLASSIC LANGCHAIN IMPORTS ---
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

# Classic Chains for Placement-Ready Logic
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

# --- PROFESSIONAL UI SETUP ---
st.set_page_config(page_title="Enterprise AI Engine", layout="wide")
st.markdown("""
    <style>
    .metric-card {
        background-color: #1e2227;
        padding: 12px;
        border-radius: 8px;
        border-left: 4px solid #00f2fe;
        margin-bottom: 10px;
        font-size: 0.9rem;
    }
    .stChatMessage { border-radius: 12px; }
    </style>
    """, unsafe_allow_html=True)

# Config
DB_PATH = "vectorstore/db_faiss"
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
llm = ChatGroq(model="llama-3.1-8b-instant")

# --- UTILS ---
def generate_pdf_summary(text):
    buffer = BytesIO()
    p = canvas.Canvas(buffer)
    p.setFont("Helvetica-Bold", 14)
    p.drawString(70, 800, "AI Analysis System: Professional Export")
    p.line(70, 790, 520, 790)
    p.setFont("Helvetica", 10)
    y = 760
    for line in text.split('\n'):
        if y < 50: p.showPage(); y = 800
        p.drawString(70, y, line[:95]); y -= 15
    p.save(); buffer.seek(0)
    return buffer

# --- SIDEBAR DASHBOARD ---
with st.sidebar:
    st.title("🛡️ Enterprise Portal")
    source = st.radio("Intelligence Vector", ["📄 Document Repository", "🌐 Web Crawler"])
    
    st.subheader("📊 System Health")
    c1, c2 = st.columns(2)
    with c1: st.markdown('<div class="metric-card"><b>Latency</b><br>Optimal</div>', unsafe_allow_html=True)
    with c2: st.markdown('<div class="metric-card"><b>Engine</b><br>Llama 3.1</div>', unsafe_allow_html=True)

    if source == "📄 Document Repository":
        files = st.file_uploader("Upload Assets", type="pdf", accept_multiple_files=True)
        process_btn = st.button("⚡ Build Index")
    else:
        url = st.text_input("Source URL")
        process_btn = st.button("🌐 Start Crawl")

    if st.button("🛑 Clear System Cache"):
        if os.path.exists("vectorstore"): shutil.rmtree("vectorstore")
        st.session_state.chat_history = []
        st.success("System Purged")
        st.rerun()

# --- DATA PROCESSING ---
if process_btn:
    docs = []
    with st.status("🛠️ Engineering Knowledge Base...", expanded=True) as status:
        if source == "📄 Document Repository" and files:
            for f in files:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(f.getvalue()); path = tmp.name
                docs.extend(PyPDFLoader(path).load()); os.remove(path)
        elif url:
            docs.extend(WebBaseLoader(url).load())
        
        if docs:
            splits = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(docs)
            vectorstore = FAISS.from_documents(splits, embeddings)
            vectorstore.save_local(DB_PATH)
            status.update(label="✅ Vector Store Synchronized", state="complete")
        else:
            st.error("Missing input data.")

# --- CONVERSATION ENGINE ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for message in st.session_state.chat_history:
    role = "user" if isinstance(message, HumanMessage) else "assistant"
    with st.chat_message(role): st.markdown(message.content)

query = st.chat_input("Enter Query...")

if query:
    with st.chat_message("user"): st.markdown(query)

    if os.path.exists(DB_PATH):
        vs = FAISS.load_local(DB_PATH, embeddings, allow_dangerous_deserialization=True)
        
        # Memory Layer 1: Contextualize
        context_prompt = ChatPromptTemplate.from_messages([
            ("system", "Rephrase the question to be a standalone query based on history."),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        retriever = create_history_aware_retriever(llm, vs.as_retriever(), context_prompt)

        # Memory Layer 2: Q&A
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", "Answer strictly using context: {context}"),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        
        chain = create_retrieval_chain(retriever, create_stuff_documents_chain(llm, qa_prompt))

        with st.spinner("Processing..."):
            res = chain.invoke({"input": query, "chat_history": st.session_state.chat_history})
            
            with st.chat_message("assistant"):
                st.markdown(res["answer"])
                st.download_button("📂 Export Analysis", generate_pdf_summary(res["answer"]), f"report_{int(time.time())}.pdf")
            
            # Update history with full context
            st.session_state.chat_history.extend([HumanMessage(content=query), AIMessage(content=res["answer"])])
    else:
        st.info("System is offline. Please index a source.")