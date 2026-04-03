from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langserve import add_routes
import uvicorn
import os
from dotenv import load_dotenv

# load env
load_dotenv()

# create app
app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="A simple API Server"
)

# ---------------- LLM (Groq) ---------------- #

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    groq_api_key=os.getenv("GROQ_API_KEY")
)

# ---------------- PROMPTS ---------------- #

prompt1 = ChatPromptTemplate.from_template(
    "Write me an essay about {topic} in 100 words"
)

prompt2 = ChatPromptTemplate.from_template(
    "Write me a poem about {topic} for a 5-year-old in 100 words"
)

# ---------------- ROUTES ---------------- #

# direct chat route
add_routes(
    app,
    llm,
    path="/chat"
)

# essay generator
add_routes(
    app,
    prompt1 | llm,
    path="/essay"
)

# poem generator
add_routes(
    app,
    prompt2 | llm,
    path="/poem"
)

# ---------------- RUN ---------------- #

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)