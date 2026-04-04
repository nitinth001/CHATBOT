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

# --- v7.0 HYBRID CYBER-TRUST UI ---
st.set_page_config(page_title="GENZ-AI PRO", layout="wide", page_icon="⚡")

st.markdown("""
    <style>
    .stApp { background: #0b0e14; color: #e0e0e0; }
    .hero-text {
        font-size: 3.5rem; font-weight: 900;
        background: linear-gradient(90deg, #00f2fe, #4facfe);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0px;
    }
    .hero-sub {
        text-align: center; color: #4facfe; margin-bottom: 30px; font-weight: 300;
    }
    .source-box {
        background: rgba(0, 242, 254, 0.05);
        border-left: 3px solid #00f2fe;
        padding: 10px; margin-top: 10px;
        font-size: 0.85rem; color: #a0a0a0;
        border-radius: 5px;
    }
    ::-webkit-scrollbar { width: 5px; }
    ::-webkit-scrollbar-thumb { background: #00f2fe; border-radius: 10px; }
    [data-testid="stSidebar"] {
        background: rgba(15, 52, 96, 0.8) !important;
        backdrop-filter: blur(10px);
        border-right: 1px solid #00f2fe;
    }
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
    p.drawString(70, 800, "GENZ-AI PRO: Official Insight Report")
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
    st.markdown("<h1 style='color: #00f2fe;'>⚡ GENZ-AI PRO</h1>", unsafe_allow_html=True)
    source = st.radio("Intelligence Mode", ["📂 Assets (PDF)", "🌐 Web Crawler"])
    
    st.markdown("---")
    if source == "📂 Assets (PDF)":
        files = st.file_uploader("Upload Knowledge Base", type="pdf", accept_multiple_files=True)
        process_btn = st.button("SYNC BASE")
    else:
        url = st.text_input("Enter Target URL")
        process_btn = st.button("CRAWL SOURCE")

    st.markdown("---")
    if st.button("HARD RESET SYSTEM"):
        if os.path.exists("vectorstore"): shutil.rmtree("vectorstore")
        st.session_state.chat_history = []
        st.success("Memory Purged")
        st.rerun()

# --- HERO SECTION ---
st.markdown('<p class="hero-text">GENZ-AI PRO</p>', unsafe_allow_html=True)
st.markdown('<p class="hero-sub">Secure AI Intelligence Engine</p>', unsafe_allow_html=True)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for message in st.session_state.chat_history:
    role = "user" if isinstance(message, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(message.content)

# Data Ingestion
if process_btn:
    docs = []
    with st.spinner("🚀 Engineering Knowledge Space..."):
        if source == "📂 Assets (PDF)" and files:
            for f in files:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(f.getvalue()); path = tmp.name
                docs.extend(PyPDFLoader(path).load()); os.remove(path)
        elif url:
            try:
                docs.extend(WebBaseLoader(url).load())
            except Exception as e:
                st.error(f"Crawl Failed: {e}")
        
        if docs:
            splits = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150).split_documents(docs)
            vectorstore = FAISS.from_documents(splits, embeddings)
            vectorstore.save_local(DB_PATH)
            st.success("Vector Store Synchronized!")
        else:
            st.error("No intelligence detected in source.")

# Interaction Layer
query = st.chat_input("Command GENZ-AI...")

if query:
    with st.chat_message("user"):
        st.markdown(query)

    # CHECK IF WE HAVE DEEP KNOWLEDGE (RAG)
    if os.path.exists(DB_PATH):
        vs = FAISS.load_local(DB_PATH, embeddings, allow_dangerous_deserialization=True)
        
        # 1. History-Aware Retriever
        context_prompt = ChatPromptTemplate.from_messages([
            ("system", "Given history and the user question, create a standalone query for search."),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        retriever = create_history_aware_retriever(llm, vs.as_retriever(search_kwargs={"k": 3}), context_prompt)

        # 2. HYBRID QA Chain: Prioritize context but allow general knowledge
        qa_system_prompt = (
            "You are GENZ-AI PRO. Follow these rules:\n"
            "1. If the provided 'context' contains the answer, use it and cite the source.\n"
            "2. If the context is irrelevant or missing the answer, use your own internal knowledge to provide a helpful response.\n"
            "3. Maintain a professional yet modern tone.\n\n"
            "Context: {context}"
        )
        
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        
        chain = create_retrieval_chain(retriever, create_stuff_documents_chain(llm, qa_prompt))

        with st.spinner("Analyzing Intelligence..."):
            res = chain.invoke({"input": query, "chat_history": st.session_state.chat_history})
            answer = res["answer"]
            
            with st.chat_message("assistant"):
                placeholder = st.empty()
                full_response = ""
                for chunk in answer.split():
                    full_response += chunk + " "
                    placeholder.markdown(full_response + "▌")
                    time.sleep(0.04)
                placeholder.markdown(full_response)
                
                # Show citations only if context was used
                if res.get("context"):
                    with st.expander("📚 View Verified Sources"):
                        for i, doc in enumerate(res["context"]):
                            source_info = doc.metadata.get('source', 'Reference')
                            page_info = f" | Page: {doc.metadata.get('page', 'N/A')}" if 'page' in doc.metadata else ""
                            st.markdown(f"**Ref {i+1}:** `{source_info}{page_info}`")
                            st.markdown(f'<div class="source-box">"{doc.page_content[:250]}..."</div>', unsafe_allow_html=True)
                
                st.download_button("💾 DL ANALYSIS", generate_pdf_summary(answer), f"GENZ_PRO_{int(time.time())}.pdf")
            
            st.session_state.chat_history.extend([HumanMessage(content=query), AIMessage(content=answer)])

    # FALLBACK: If no PDF is uploaded, act as a standard Chatbot
    else:
        with st.chat_message("assistant"):
            chat_prompt = ChatPromptTemplate.from_messages([
                ("system", "You are GENZ-AI PRO, a helpful AI. No external files are uploaded currently, so use your own knowledge."),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ])
            # Direct chain without retrieval
            response = llm.invoke(chat_prompt.format_messages(input=query, chat_history=st.session_state.chat_history))
            
            placeholder = st.empty()
            full_response = ""
            for chunk in response.content.split():
                full_response += chunk + " "
                placeholder.markdown(full_response + "▌")
                time.sleep(0.04)
            placeholder.markdown(full_response)
            
            st.session_state.chat_history.extend([HumanMessage(content=query), AIMessage(content=response.content)])

# Status Footer
if not os.path.exists(DB_PATH):
    st.info("💡 GENZ-AI is in General Mode. Sync a Knowledge Base for Deep Research.")