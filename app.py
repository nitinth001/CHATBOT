# imports
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

import streamlit as st
from dotenv import load_dotenv
import os

# load env
load_dotenv()

# create model (Groq)
llm = ChatGroq(
    model="llama-3.1-8b-instant",   # fast + free
    groq_api_key=os.getenv("GROQ_API_KEY")
)

# prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("user", "{input}")
])

# parser
output_parser = StrOutputParser()

# chain
chain = prompt | llm | output_parser

# UI
st.title("Free Chatbot (Groq)")

user_input = st.text_input("Ask something:")

if user_input:
    response = chain.invoke({"input": user_input})
    st.write(response)