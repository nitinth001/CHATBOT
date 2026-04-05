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
# 🔐 LOAD ENV VARIABLES
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
# 🤖 LOAD MODELS
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
# 📄 PDF GENERATOR (IMPROVED)
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
# 🧠 VECTOR STORE HANDLING
# ==============================
def create_vectorstore(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    splits = splitter.split_documents(docs)

    vs = FAISS.from_documents(splits, embeddings)
    vs.save_local(DB_PATH)


def load_vectorstore():
    if os.path.exists(DB_PATH):
        return FAISS.load_local(
            DB_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )
    return None

# ==============================
# 🎨 UI
# ==============================
st.title("😎 GENZ-AI")
st.caption("Next Gen AI Intelligence System")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ==============================
# 📂 SIDEBAR
# ==============================
with st.sidebar:
    st.header("⚙️ Control Panel")

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
        st.success("System Reset Done")
        st.rerun()

# ==============================
# 📥 DATA INGESTION
# ==============================
if process:
    docs = []

    with st.spinner("Processing..."):
        try:
            if source == "PDF" and files:
                for f in files:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(f.read())
                        path = tmp.name

                    docs.extend(PyPDFLoader(path).load())
                    os.remove(path)

            elif source == "Web" and url:
                docs.extend(WebBaseLoader(url).load())

            if docs:
                create_vectorstore(docs)
                st.success("Vector DB Created ✅")
            else:
                st.warning("No data found")

        except Exception as e:
            st.error(f"Error: {str(e)}")

# ==============================
# 💬 CHAT SYSTEM
# ==============================
query = st.chat_input("Ask something...")

if query:
    st.chat_message("user").markdown(query)

    vs = load_vectorstore()

    # ==========================
    # 🔍 RAG MODE
    # ==========================
    if vs:
        retriever_prompt = ChatPromptTemplate.from_messages([
            ("system", "Convert conversation into standalone query"),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ])

        retriever = create_history_aware_retriever(
            llm,
            vs.as_retriever(search_kwargs={"k": 3}),
            retriever_prompt
        )

        qa_prompt = ChatPromptTemplate.from_messages([
            ("system",
             "Use context if relevant. Otherwise answer normally.\nContext: {context}"),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ])

        chain = create_retrieval_chain(
            retriever,
            create_stuff_documents_chain(llm, qa_prompt)
        )

        with st.spinner("Thinking..."):
            res = chain.invoke({
                "input": query,
                "chat_history": st.session_state.chat_history
            })

            answer = res["answer"]

    # ==========================
    # 🤖 NORMAL MODE
    # ==========================
    else:
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful AI assistant"),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ])

        response = llm.invoke(
            prompt.format_messages(
                input=query,
                chat_history=st.session_state.chat_history
            )
        )
        answer = response.content

    # ==========================
    # 🖥️ STREAM OUTPUT
    # ==========================
    with st.chat_message("assistant"):
        msg = st.empty()
        full = ""

        for word in answer.split():
            full += word + " "
            msg.markdown(full + "▌")
            time.sleep(0.02)

        msg.markdown(full)

        st.download_button(
            "📄 Download Report",
            generate_pdf_summary(answer),
            "report.pdf"
        )

    # Save history
    st.session_state.chat_history.extend([
        HumanMessage(content=query),
        AIMessage(content=answer)
    ])

# ==============================
# 🧾 FOOTER
# ==============================
if not os.path.exists(DB_PATH):
    st.info("ℹ️ Running in General Mode (No Documents Loaded)")