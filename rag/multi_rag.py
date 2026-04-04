import streamlit as st
import os
import tempfile
import shutil
import time
from io import BytesIO
from dotenv import load_dotenv
from reportlab.pdfgen import canvas

# Classic LangChain Imports
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

# --- v5.0 CHATGPT-STYLE CSS ---
st.set_page_config(page_title="Nexus AI Engine", layout="wide", page_icon="🤖")

st.markdown("""
    <style>
    /* Main Background */
    .stApp { background-color: #212121; }
    
    /* ChatGPT-style Chat Bubbles */
    .stChatMessage {
        background-color: #2f2f2f !important;
        border: 1px solid #424242 !important;
        border-radius: 15px !important;
        padding: 15px !important;
        margin-bottom: 20px !important;
        max-width: 85% !important;
    }
    
    /* Metrics glassmorphism */
    .metric-container {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 10px;
        padding: 15px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        text-align: center;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Professional Sidebar */
    [data-testid="stSidebar"] {
        background-color: #171717 !important;
        border-right: 1px solid #303030;
    }
    </style>
    """, unsafe_allow_html=True)

# Persistent Config
DB_PATH = "vectorstore/db_faiss"
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
llm = ChatGroq(model="llama-3.1-8b-instant")

# --- UTILS ---
def generate_pdf_summary(text):
    buffer = BytesIO()
    p = canvas.Canvas(buffer)
    p.setFont("Helvetica-Bold", 14)
    p.drawString(70, 800, "Nexus AI: Executive Summary")
    p.line(70, 790, 520, 790)
    p.setFont("Helvetica", 10)
    y = 760
    for line in text.split('\n'):
        if y < 50: p.showPage(); y = 800
        p.drawString(70, y, line[:95]); y -= 15
    p.save(); buffer.seek(0)
    return buffer

# --- SIDEBAR: PRODUCT CONFIG ---
with st.sidebar:
    st.title("🤖 Nexus AI")
    st.caption("v5.0 Enterprise Edition")
    
    source = st.selectbox("Intelligence Mode", ["Document PDF", "Web URL Crawler"])
    
    st.markdown("---")
    if source == "Document PDF":
        files = st.file_uploader("Upload Knowledge Base", type="pdf", accept_multiple_files=True)
        process_btn = st.button("Initialize Engine", use_container_width=True)
    else:
        url = st.text_input("Source URL")
        process_btn = st.button("Crawl Intelligence", use_container_width=True)

    st.markdown("---")
    st.subheader("📊 System Diagnostics")
    c1, c2 = st.columns(2)
    with c1: st.markdown('<div class="metric-container"><small>Memory</small><br><b>Active</b></div>', unsafe_allow_html=True)
    with c2: st.markdown('<div class="metric-container"><small>Latency</small><br><b>14ms</b></div>', unsafe_allow_html=True)

    if st.sidebar.button("Clear Session History", use_container_width=True):
        if os.path.exists("vectorstore"): shutil.rmtree("vectorstore")
        st.session_state.chat_history = []
        st.rerun()

# --- CHAT INTERFACE ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Title at the top center
st.markdown("<h2 style='text-align: center; color: white;'>How can I help you today?</h2>", unsafe_allow_html=True)

# Display Chat History
for message in st.session_state.chat_history:
    role = "user" if isinstance(message, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(message.content)

# Data Processing
if process_btn:
    docs = []
    with st.spinner("Processing knowledge vectors..."):
        if source == "Document PDF" and files:
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
            st.success("Indexing Synchronized.")
        else:
            st.error("Input source empty.")

# User Input
query = st.chat_input("Message Nexus AI...")

if query:
    with st.chat_message("user"):
        st.markdown(query)

    if os.path.exists(DB_PATH):
        vs = FAISS.load_local(DB_PATH, embeddings, allow_dangerous_deserialization=True)
        
        # Memory Retriever
        context_prompt = ChatPromptTemplate.from_messages([
            ("system", "Given history and question, create a standalone query."),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        retriever = create_history_aware_retriever(llm, vs.as_retriever(), context_prompt)

        # Q&A Logic
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", "Answer based on context: {context}"),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        
        chain = create_retrieval_chain(retriever, create_stuff_documents_chain(llm, qa_prompt))

        with st.spinner(""): # Minimalist thinking indicator
            res = chain.invoke({"input": query, "chat_history": st.session_state.chat_history})
            
            with st.chat_message("assistant"):
                st.markdown(res["answer"])
                # Sleek download button
                st.download_button("💾 Save Export", generate_pdf_summary(res["answer"]), f"Nexus_{int(time.time())}.pdf")
            
            st.session_state.chat_history.extend([HumanMessage(content=query), AIMessage(content=res["answer"])])
    else:
        st.info("Please initialize the knowledge base in the sidebar.")