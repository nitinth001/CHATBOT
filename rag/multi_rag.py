import streamlit as st
import os
import tempfile
import shutil
import time
from io import BytesIO
from dotenv import load_dotenv
from reportlab.pdfgen import canvas

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

# --- v4.0 PROFESSIONAL UI SETUP ---
st.set_page_config(page_title="Enterprise RAG Solution", layout="wide")
st.markdown("""
    <style>
    .metric-card {
        background-color: #1e2227;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #00f2fe;
        margin-bottom: 10px;
    }
    .stChatMessage { border-radius: 15px; margin-bottom: 10px; }
    </style>
    """, unsafe_allow_html=True)

# Persistent Config
DB_PATH = "vectorstore/db_faiss"
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
llm = ChatGroq(model="llama-3.1-8b-instant", groq_api_key=os.getenv("GROQ_API_KEY"))

# --- UTILS ---
def generate_pdf_summary(text):
    buffer = BytesIO()
    p = canvas.Canvas(buffer)
    p.drawString(100, 800, "Official AI Analysis Export")
    p.line(100, 785, 500, 785)
    y = 760
    for line in text.split('\n'):
        if y < 50: p.showPage(); y = 800
        p.drawString(100, y, line[:90]); y -= 15
    p.save(); buffer.seek(0)
    return buffer

# --- SIDEBAR DASHBOARD ---
with st.sidebar:
    st.title("⚙️ System Control")
    source_type = st.radio("Intelligence Source", ["📄 PDF Documents", "🔗 Web Intelligence"])
    
    # UI Metrics (The "Professional" Touch)
    st.subheader("📊 System Metrics")
    m1, m2 = st.columns(2)
    with m1:
        st.markdown('<div class="metric-card"><b>Status</b><br>Online</div>', unsafe_allow_html=True)
    with m2:
        st.markdown(f'<div class="metric-card"><b>Model</b><br>Llama3.1</div>', unsafe_allow_html=True)

    if source_type == "📄 PDF Documents":
        files = st.file_uploader("Upload Assets", type="pdf", accept_multiple_files=True)
        process_btn = st.button("⚡ Build Knowledge Base")
    else:
        url = st.text_input("Source URL")
        process_btn = st.button("🌐 Scrape & Index")

    if st.button("🛑 Factory Reset"):
        if os.path.exists("vectorstore"): shutil.rmtree("vectorstore")
        st.session_state.chat_history = []
        st.rerun()

# --- v4.0 CORE LOGIC: CONVERSATIONAL RAG ---
if process_btn:
    docs = []
    with st.status("🚀 Engineering Vector Space...", expanded=True) as status:
        if source_type == "📄 PDF Documents" and files:
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
            status.update(label="✅ Indexing Complete!", state="complete")
        else:
            st.error("No data found.")

# --- CONVERSATION MEMORY STATE ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display History
for message in st.session_state.chat_history:
    role = "user" if isinstance(message, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(message.content)

# Chat Input
query = st.chat_input("Command the AI...")

if query:
    with st.chat_message("user"):
        st.markdown(query)

    if os.path.exists(DB_PATH):
        vs = FAISS.load_local(DB_PATH, embeddings, allow_dangerous_deserialization=True)
        
        # 1. Contextualize Question (The "Memory" Part)
        context_q_system_prompt = "Given a chat history and the latest user question, formulate a standalone question."
        context_q_prompt = ChatPromptTemplate.from_messages([
            ("system", context_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        history_aware_retriever = create_history_aware_retriever(llm, vs.as_retriever(), context_q_prompt)

        # 2. Answer Question
        qa_system_prompt = "Use the context to answer. Context: {context}"
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        
        qa_chain = create_stuff_documents_chain(llm, qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain)

        # Execute
        res = rag_chain.invoke({"input": query, "chat_history": st.session_state.chat_history})
        
        with st.chat_message("assistant"):
            st.markdown(res["answer"])
            st.download_button("📂 Export Analysis", generate_pdf_summary(res["answer"]), f"analysis_{int(time.time())}.pdf")
        
        # Update History
        st.session_state.chat_history.extend([HumanMessage(content=query), AIMessage(content=res["answer"])])
    else:
        st.info("System idle. Please provide a data source.")