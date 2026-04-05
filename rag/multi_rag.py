import streamlit as st
import os
import tempfile
import shutil
import time
from io import BytesIO
from dotenv import load_dotenv

# PDF
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

# LangChain
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
st.set_page_config(page_title="GENZ-AI", layout="wide")
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
    font-size: 3.2rem;
    font-weight: 800;
    text-align: center;
    background: linear-gradient(90deg, #38bdf8, #6366f1, #8b5cf6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.hero-sub {
    text-align: center;
    color: #94a3b8;
}
[data-testid="stSidebar"] {
    background: rgba(15, 23, 42, 0.7);
    backdrop-filter: blur(20px);
}
.stChatInputContainer {
    background: rgba(15, 23, 42, 0.7) !important;
    backdrop-filter: blur(12px);
    border-radius: 20px;
}
header, footer {visibility: hidden;}
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
# 📄 PDF
# ==============================
def generate_pdf_summary(text):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer)
    styles = getSampleStyleSheet()

    content = []
    content.append(Paragraph("GENZ-AI Insight Report", styles["Title"]))
    content.append(Spacer(1, 12))

    for line in text.split("\n"):
        content.append(Paragraph(line, styles["Normal"]))
        content.append(Spacer(1, 8))

    doc.build(content)
    buffer.seek(0)
    return buffer

# ==============================
# 🧠 VECTOR STORE
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
# 🧠 PROMPT (FIXED)
# ==============================
def get_prompt():
    return ChatPromptTemplate.from_messages([
        ("system",
         "You are an AI tutor.\n\n"
         "For every question:\n"
         "1. First explain in simple words.\n"
         "2. Then give Python code.\n"
         "3. ALWAYS format code like:\n"
         "```python\ncode here\n```\n"
         "4. Separate each question clearly.\n"
        ),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])

# ==============================
# 🧹 CLEAN RESPONSE
# ==============================
def clean_response(text):
    text = text.replace("python def", "```python\ndef")
    if "```python" in text and text.count("```") % 2 != 0:
        text += "\n```"
    return text

# ==============================
# 🎯 HEADER
# ==============================
st.markdown('<p class="hero-text">🤖 GENZ-AI</p>', unsafe_allow_html=True)
st.markdown('<p class="hero-sub">AI Tutor + Research Assistant</p>', unsafe_allow_html=True)
st.markdown("<div style='text-align:center; color:#22c55e;'>● Online</div>", unsafe_allow_html=True)

# ==============================
# 💬 SESSION
# ==============================
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ==============================
# 📂 SIDEBAR
# ==============================
with st.sidebar:
    st.header("Control Panel")

    source = st.radio("Select Source", ["PDF", "Web"])

    if source == "PDF":
        files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
    else:
        url = st.text_input("Enter URL")

    process = st.button("Process Data")

    if st.button("Reset System"):
        if os.path.exists("vectorstore"):
            shutil.rmtree("vectorstore")
        st.session_state.chat_history = []
        st.rerun()

# ==============================
# 📥 INGEST
# ==============================
if process:
    docs = []
    with st.spinner("Processing..."):
        try:
            if source == "PDF" and files:
                for f in files:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(f.read())
                        docs.extend(PyPDFLoader(tmp.name).load())
                        os.remove(tmp.name)

            elif source == "Web" and url:
                docs.extend(WebBaseLoader(url).load())

            if docs:
                create_vectorstore(docs)
                st.success("Knowledge Base Ready")
            else:
                st.warning("No data found")

        except Exception as e:
            st.error(str(e))

# ==============================
# 💬 DISPLAY CHAT
# ==============================
for message in st.session_state.chat_history:
    role = "user" if isinstance(message, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(message.content)

# ==============================
# 💬 CHAT
# ==============================
query = st.chat_input("Ask anything...")

if query:
    st.chat_message("user").markdown(query)

    vs = load_vectorstore()
    prompt = get_prompt()

    if vs:
        retriever = create_history_aware_retriever(
            llm,
            vs.as_retriever(search_kwargs={"k": 3}),
            prompt
        )

        chain = create_retrieval_chain(
            retriever,
            create_stuff_documents_chain(llm, prompt)
        )

        res = chain.invoke({
            "input": query,
            "chat_history": st.session_state.chat_history
        })

        answer = res["answer"]

    else:
        response = llm.invoke(
            prompt.format_messages(
                input=query,
                chat_history=st.session_state.chat_history
            )
        )
        answer = response.content

    # ✅ FIX OUTPUT
    answer = clean_response(answer)

    # ==========================
    # ✨ STREAM OUTPUT
    # ==========================
    with st.chat_message("assistant"):
        msg = st.empty()
        full = ""

        for word in answer.split():
            full += word + " "
            msg.markdown(full + "▌", unsafe_allow_html=True)
            time.sleep(0.02)

        msg.markdown(full, unsafe_allow_html=True)

        st.download_button(
            "📄 Download Report",
            generate_pdf_summary(answer),
            "report.pdf"
        )

    st.session_state.chat_history.extend([
        HumanMessage(content=query),
        AIMessage(content=answer)
    ])

# ==============================
# 📢 FOOTER
# ==============================
if not os.path.exists(DB_PATH):
    st.info("💡 Running in General Mode")