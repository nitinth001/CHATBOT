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

# --- v5.1 NEON CYBERPUNK UI ---
st.set_page_config(page_title="GENZ-AI", layout="wide", page_icon="⚡")

st.markdown("""
    <style>
    /* Main Background & Fonts */
    .stApp { 
        background: radial-gradient(circle at top right, #1a1a2e, #16213e, #0f3460);
        color: #e94560;
    }
    
    /* Neon Title Gradient */
    .hero-text {
        font-size: 3.5rem;
        font-weight: 800;
        background: -webkit-linear-gradient(#00f2fe, #4facfe);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0px;
    }

    /* Glassmorphism Sidebar */
    [data-testid="stSidebar"] {
        background: rgba(15, 52, 96, 0.8) !important;
        backdrop-filter: blur(10px);
        border-right: 1px solid #00f2fe;
    }

    /* Glowing Chat Bubbles */
    .stChatMessage {
        background: rgba(255, 255, 255, 0.05) !important;
        border: 1px solid rgba(0, 242, 254, 0.3) !important;
        border-radius: 20px !important;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
        margin-bottom: 15px !important;
    }

    /* Buttons with Neon Glow */
    .stButton>button {
        width: 100%;
        border-radius: 25px !important;
        border: 1px solid #00f2fe !important;
        background: transparent !important;
        color: #00f2fe !important;
        font-weight: bold !important;
        transition: 0.4s;
    }
    .stButton>button:hover {
        background: #00f2fe !important;
        color: #1a1a2e !important;
        box-shadow: 0 0 20px #00f2fe;
    }

    /* Chat Input Styling */
    .stChatInputContainer {
        padding-bottom: 20px !important;
    }

    /* Hide redundant elements */
    header, footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

# Configuration
DB_PATH = "vectorstore/db_faiss"
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
llm = ChatGroq(model="llama-3.1-8b-instant")

# --- UTILS ---
def generate_pdf_summary(text):
    buffer = BytesIO()
    p = canvas.Canvas(buffer)
    p.setFont("Helvetica-Bold", 14)
    p.drawString(70, 800, "GENZ-AI: Data Insight Report")
    p.line(70, 790, 520, 790)
    p.setFont("Helvetica", 10)
    y = 760
    for line in text.split('\n'):
        if y < 50: p.showPage(); y = 800
        p.drawString(70, y, line[:95]); y -= 15
    p.save(); buffer.seek(0)
    return buffer

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("<h1 style='color: #00f2fe;'>⚡ GENZ-AI</h1>", unsafe_allow_html=True)
    st.caption("Pushing RAG to the limit.")
    
    source = st.radio("Select Input", ["📂 PDF Files", "🌐 Live Link"])
    
    st.markdown("---")
    if source == "📂 PDF Files":
        files = st.file_uploader("Drop Files", type="pdf", accept_multiple_files=True)
        process_btn = st.button("SYNC KNOWLEDGE")
    else:
        url = st.text_input("URL Link")
        process_btn = st.button("CRAWL WEB")

    if st.button("PURGE MEMORY"):
        if os.path.exists("vectorstore"): shutil.rmtree("vectorstore")
        st.session_state.chat_history = []
        st.rerun()

# --- HERO SECTION ---
st.markdown('<p class="hero-text">GENZ-AI</p>', unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #4facfe;'>What's the move today?</p>", unsafe_allow_html=True)

# Chat State
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for message in st.session_state.chat_history:
    role = "user" if isinstance(message, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(message.content)

# Data Processing
if process_btn:
    docs = []
    with st.spinner("Processing..."):
        if source == "📂 PDF Files" and files:
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
            st.success("System Synchronized.")
        else:
            st.error("Target source is empty.")

# Interaction
query = st.chat_input("Talk to GENZ-AI...")

if query:
    with st.chat_message("user"):
        st.markdown(query)

    if os.path.exists(DB_PATH):
        vs = FAISS.load_local(DB_PATH, embeddings, allow_dangerous_deserialization=True)
        
        # Memory-Enhanced Retrieval
        context_prompt = ChatPromptTemplate.from_messages([
            ("system", "Keep it cool and accurate. Contextualize the prompt based on history."),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        retriever = create_history_aware_retriever(llm, vs.as_retriever(), context_prompt)

        # Response Chain
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are GENZ-AI. Use the context: {context}"),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        
        chain = create_retrieval_chain(retriever, create_stuff_documents_chain(llm, qa_prompt))

        with st.spinner(""): 
            res = chain.invoke({"input": query, "chat_history": st.session_state.chat_history})
            
            with st.chat_message("assistant"):
                st.markdown(res["answer"])
                st.download_button("💾 DL ANALYSIS", generate_pdf_summary(res["answer"]), f"GENZ_{int(time.time())}.pdf")
            
            st.session_state.chat_history.extend([HumanMessage(content=query), AIMessage(content=res["answer"])])
    else:
        st.info("Feed me some data to get started.")