import requests
import streamlit as st

# -------- API CALLS -------- #

def get_essay_response(input_text):
    response = requests.post(
        "http://localhost:8000/essay/invoke",
        json={'input': {'topic': input_text}}
    )
    return response.json()['output']['content']


def get_poem_response(input_text):
    response = requests.post(
        "http://localhost:8000/poem/invoke",
        json={'input': {'topic': input_text}}
    )
    return response.json()['output']['content']


# -------- STREAMLIT UI -------- #

st.title('LangChain Demo (Groq API)')

essay_input = st.text_input("Write an essay on:")
poem_input = st.text_input("Write a poem on:")

if essay_input:
    st.write(get_essay_response(essay_input))

if poem_input:
    st.write(get_poem_response(poem_input))