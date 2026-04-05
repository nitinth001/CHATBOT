import streamlit as st
import os
import tempfile
import shutil
import time
from io import BytesIO
from dotenv import load_dotenv

# PDF Generation
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

# LangChain - Preserving your requested "Classic" chains
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# ==============================
# 🔐 ENV SETUP
# ==============================
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.error("❌ GROQ API KEY missing. Add it in .env file")
    st.stop()

# ==============================
# ⚙️ CONFIG
# ==============================
st.set_page_config(page_title="GENZ-AI | Universal", layout="wide")
DB_PATH = "vectorstore/db_faiss"

# ==============================
# 🎨 MODERN UI
# ==============================
st.markdown("""
<style>
.stApp {
    background: radial-gradient(circle at top, #0f172a, #020617);
    color: #e2e8f0;
    font-family: 'Inter', sans-serif;
}
.hero-text {
    font-size: 3rem;
    font-weight: 800;
    text-align: center;
    background: linear-gradient(90deg, #38bdf8, #6366f1);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0px;
}
.hero-sub {
    text-align: center;
    color: #94a3b8;
    margin-bottom: 2rem;
}
[data-testid="stSidebar"] {
    background: rgba(15, 23, 42, 0.7);
    backdrop-filter: blur(20px);
}
</style>
""", unsafe_allow_html=True)

# ==============================
# 🤖 MODELS
# ==============================
@st.cache_resource
def load_llm():
    return ChatGroq(
        model="llama-3.1-8b-instant",
        groq_api_key=GROQ_API_KEY
    )

@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

llm = load_llm()
embeddings = load_embeddings()

# ==============================
# 📄 PDF GENERATOR
# ==============================
def generate_pdf_summary(text):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer)
    styles = getSampleStyleSheet()

    content = []
    content.append(Paragraph("GENZ-AI Insights Report", styles["Title"]))
    content.append(Spacer(1, 12))

    for line in text.split("\n"):
        if line.strip():
            # Basic cleanup for reportlab
            clean_line = line.replace('*', '').replace('#', '')
            content.append(Paragraph(clean_line, styles["Normal"]))
            content.append(Spacer(1, 8))

    doc.build(content)
    buffer.seek(0)
    return buffer

# ==============================
# 🧠 VECTOR STORE LOGIC
# ==============================
def create_vectorstore(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    splits = splitter.split_documents(docs)
    vs = FAISS.from_documents(splits, embeddings)
    vs.save_local(DB_PATH)

def load_vectorstore():
    if os.path.exists(DB_PATH):
        return FAISS.load_local(DB_PATH, embeddings, allow_dangerous_deserialization=True)
    return None

# ==============================
# 🧠 GENERALIZED PROMPTS
# ==============================
def get_rag_prompt():
    return ChatPromptTemplate.from_messages([
        ("system",
         "You are an expert AI assistant. Use the provided context to answer the user's request.\n\n"
         "Context:\n{context}\n\n"
         "Instructions:\n"
         "1. If the query requires calculations or logic, show step-by-step work.\n"
         "2. Provide Python code examples using ```python blocks where helpful.\n"
         "3. If the answer is not in the context, use your general knowledge but clarify that it's not in the documents."
        ),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])

def get_normal_prompt():
    return ChatPromptTemplate.from_messages([
        ("system",
         "You are a helpful and witty AI assistant. Provide clear explanations and code examples when relevant."
        ),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])

# ==============================
# 🎯 HEADER
# ==============================
st.markdown('<p class="hero-text">GENZ-AI</p>', unsafe_allow_html=True)
st.markdown('<p class="hero-sub">Universal AI Assistant: PDF Research • Web Analysis • General Chat</p>', unsafe_allow_html=True)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ==============================
# 📂 SIDEBAR (Source Selection)
# ==============================
with st.sidebar:
    st.header("🛠️ Configuration")
    source_type = st.selectbox("Intelligence Source", ["None (General Chat)", "PDF Upload", "Web URL"])

    docs = []
    if source_type == "PDF Upload":
        files = st.file_uploader("Upload Documents", type="pdf", accept_multiple_files=True)
        if st.button("Index PDFs"):
            with st.spinner("Reading PDFs..."):
                for f in files:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(f.read())
                        docs.extend(PyPDFLoader(tmp.name).load())
                        os.remove(tmp.name)
                if docs:
                    create_vectorstore(docs)
                    st.success("Vector Database Created!")

    elif source_type == "Web URL":
        url = st.text_input("Enter URL (e.g., article, blog, docs)")
        if st.button("Fetch Content"):
            with st.spinner("Scraping Webpage..."):
                try:
                    docs.extend(WebBaseLoader(url).load())
                    create_vectorstore(docs)
                    st.success("Web Content Indexed!")
                except Exception as e:
                    st.error(f"Error: {e}")

    st.markdown("---")
    if st.button("Reset Everything"):
        if os.path.exists("vectorstore"):
            shutil.rmtree("vectorstore")
        st.session_state.chat_history = []
        st.rerun()

# ==============================
# 💬 CHAT DISPLAY
# ==============================
for message in st.session_state.chat_history:
    role = "user" if isinstance(message, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(message.content)

# ==============================
# 💬 CHAT LOGIC
# ==============================
query = st.chat_input("How can I help you today?")

if query:
    st.chat_message("user").markdown(query)
    vs = load_vectorstore()
    
    # Check if we should use RAG or Normal LLM logic
    if vs and source_type != "None (General Chat)":
        # 1. RAG CHAIN LOGIC (PREVIOUS LOGIC PRESERVED)
        retriever_prompt = ChatPromptTemplate.from_messages([
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
            ("system", "Generate a search query to look up the relevant context for the user request.")
        ])
        
        retriever = create_history_aware_retriever(llm, vs.as_retriever(search_kwargs={"k": 5}), retriever_prompt)
        document_chain = create_stuff_documents_chain(llm, get_rag_prompt())
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        with st.spinner("🧐 Analyzing sources..."):
            result = retrieval_chain.invoke({
                "input": query,
                "chat_history": st.session_state.chat_history
            })
            answer = result["answer"]
    else:
        # 2. NORMAL MODE (GENERAL SEARCHING/CHAT)
        with st.spinner("💭 Thinking..."):
            chain = get_normal_prompt() | llm
            response = chain.invoke({
                "input": query, 
                "chat_history": st.session_state.chat_history
            })
            answer = response.content

    # ==========================
    # ✨ DISPLAY & DOWNLOAD
    # ==========================
    with st.chat_message("assistant"):
        st.markdown(answer)
        st.download_button(
            "📥 Download Response as PDF",
            generate_pdf_summary(answer),
            f"genzai_response_{int(time.time())}.pdf"
        )

    st.session_state.chat_history.extend([
        HumanMessage(content=query),
        AIMessage(content=answer)
    ])