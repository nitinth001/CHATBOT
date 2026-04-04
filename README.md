# ⚡ GENZ-AI PRO: Hybrid Intelligence Engine

**GENZ-AI PRO** is a high-performance, enterprise-grade RAG (Retrieval-Augmented Generation) application built to eliminate AI hallucinations by grounding LLM responses in verified private data.

![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)

## 🚀 Live Demo
[Insert Your Streamlit Link Here]

---

## 🌟 Key Features

- **Hybrid Intelligence:** Seamlessly switches between **Deep Research Mode** (using uploaded PDFs/URLs) and **General Mode** (using Llama 3.1's internal knowledge).
- **History-Aware Retrieval:** Maintains full conversational context. The AI understands pronouns and follow-up questions by re-contextualizing queries based on chat history.
- **Verifiable Citations:** Includes a "Verified Sources" engine that maps AI answers back to specific document metadata and page numbers for 100% transparency.
- **High-Speed Inference:** Powered by **Groq LPU™** technology, delivering responses at sub-second speeds.
- **Cyber-Trust UI:** A custom-styled, neon-themed interface with real-time response streaming.

---

## 🛠️ The Tech Stack

| Layer | Technology |
| :--- | :--- |
| **Frontend** | Streamlit (Custom CSS) |
| **LLM** | Meta Llama 3.1 (via Groq API) |
| **Orchestration** | LangChain |
| **Vector DB** | FAISS (Facebook AI Similarity Search) |
| **Embeddings** | HuggingFace (all-MiniLM-L6-v2) |
| **Deployment** | Docker & Streamlit Cloud |

---

## 🧠 How It Works (RAG Architecture)



1. **Ingestion:** Documents (PDFs) or Web URLs are loaded and split into semantic chunks.
2. **Vectorization:** Text chunks are converted into high-dimensional vectors using HuggingFace embeddings.
3. **Storage:** Vectors are stored in a local FAISS index for lightning-fast similarity searching.
4. **Retrieval:** When a user asks a question, the system finds the top "k" relevant chunks and passes them to the LLM as context.
5. **Generation:** The LLM generates a response grounded strictly in the retrieved data.

---

## 🛠️ Installation & Setup

1. **Clone the repo:**
   ```bash
   git clone [https://github.com/YOUR_USERNAME/CHATBOT.git](https://github.com/YOUR_USERNAME/CHATBOT.git)
   cd CHATBOT


Install Dependencies:

Bash
pip install -r requirements.txt
Set up Environment Variables:
Create a .env file and add your Groq API Key:

Code snippet
GROQ_API_KEY=your_api_key_here
Run the App:

Bash
streamlit run multi_rag.py
🐳 Docker Support
This project is container-ready. To build and run using Docker:

Bash
docker build -t genz-ai-pro .
docker run -p 8501:8501 genz-ai-pro
👨‍💻 Author
Nitin AI Enthusiast & Full-Stack Developer


### 💡 Pro-Tips for your README:
1. **The Link:** Replace `[Insert Your Streamlit Link Here]` with the actual link to your app.
2. **The Username:** In the Installation section, make sure the `git clone` link matches your actual GitHub username.
3. **Images:** If you want to be extra fancy, take a screenshot of your app and upload it to the GitHub repo, then link it in the README so people can see the "Neon UI" before they even click the link.

**Once you push this README, your GitHub will look 100% professional. Ready to s   
