# Step 1: Start with a clean Python 3.11 environment
FROM python:3.11-slim

# Step 2: Create a folder called /app inside the container to hold our code
WORKDIR /app

# Step 3: Install essential system tools that FAISS and PDF loaders need
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Step 4: Copy your requirements.txt first (this makes building faster)
COPY requirements.txt .

# Step 5: Install all the libraries (LangChain, Groq, Streamlit, etc.)
RUN pip install --no-cache-dir -r requirements.txt

# Step 6: Copy every other file from your folder (multi_rag.py, .env, etc.) into the container
COPY . .

# Step 7: Tell the container to listen on port 8501 (Streamlit's default port)
EXPOSE 8501

# Step 8: The command to start your GENZ-AI when the container "wakes up"
ENTRYPOINT ["streamlit", "run", "multi_rag.py", "--server.port=8501", "--server.address=0.0.0.0"]
