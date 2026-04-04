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
    .stButton>button { 
        border-radius: 20px; 
        border: 1px solid #ff4b4b; 
        background-color: #0e1117; 
        color: white; 
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #ff4b4b;
        color: white;
    }
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
    p.setFont("Helvetica-Bold", 16)
    p.drawString(100, 800, "AI Generated Analysis")
    p.line(100, 790, 500, 790)
    
    p.setFont("Helvetica", 10)
    y = 760
    # Simple wrap and draw
    for line in text.split('\n'):
        if y < 50:
            p.showPage()
            y = 800
        # Prevent long lines from breaking the PDF
        chunk_size = 85
        chunks = [line[i:i+chunk_size] for i in range(0, len(line), chunk_size)]
        for chunk in chunks:
            p.drawString(100, y, chunk)
            y -= 15
    p.save()
    buffer.seek(0)
    return buffer

st.title("🚀 AI PDF Pro v2.0")

# --- Sidebar ---
with st.sidebar:
    st.header("📂 Document Lab")
    uploaded_files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
    process_btn = st.button("🚀 Index Everything")
    
    st.divider()
    if st.button("🗑️ Reset Knowledge Base"):
        if os.path.exists("vectorstore"):
            shutil.rmtree("vectorstore")
        st.session_state.messages = []
        st.success("System Reset!")
        st.rerun()

# --- Logic: Processing Multiple PDFs ---
if uploaded_files and process_btn:
    all_docs = []
    with st.spinner("Analyzing Documents..."):
        for uploaded_file in uploaded_files:
            # Securely handle temp files to avoid EmptyFileError
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name
            
            try:
                loader = PyPDFLoader(tmp_path)
                all_docs.extend(loader.load())
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(all_docs)
        
        vectorstore = FAISS.from_documents(splits, embeddings)
        vectorstore.save_local(DB_PATH)
        st.success(f"Successfully Indexed {len(uploaded_files)} PDF(s)!")

# --- Chat Interface ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Ask a question about your documents...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    if os.path.exists(DB_PATH):
        vectorstore = FAISS.load_local(DB_PATH, embeddings, allow_dangerous_deserialization=True)
        
        # PROMPT with Citation instructions
        prompt = ChatPromptTemplate.from_template("""
        Answer accurately using the context. 
        At the end of your answer, list the source page numbers.
        Context: {context}
        Question: {input}""")
        
        combine_chain = create_stuff_documents_chain(llm, prompt)
        retrieval_chain = create_retrieval_chain(vectorstore.as_retriever(), combine_chain)
        
        with st.spinner("Thinking..."):
            response = retrieval_chain.invoke({"input": user_input})
            answer = response['answer']
            
            # Extract page numbers from metadata
            pages = set([str(doc.metadata.get('page', 0) + 1) for doc in response['context']])
            citation_text = f"\n\n**Sources:** Page(s) {', '.join(sorted(pages))}"
            full_response = answer + citation_text

            st.session_state.messages.append({"role": "assistant", "content": full_response})
            with st.chat_message("assistant"):
                st.markdown(full_response)
                
                # Download Button for the specific response
                pdf_file = generate_pdf_summary(full_response)
                st.download_button(
                    label="📥 Download this Answer as PDF",
                    data=pdf_file,
                    file_name="ai_analysis.pdf",
                    mime="application/pdf"
                )
    else:
        st.info("Please upload and index a PDF to activate the AI.")