import os
import streamlit as st
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Prompt Template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Please respond to the question asked."),
        ("user", "Question: {question}")
    ]
)

# Streamlit Framework
st.title("LangChain Demo with Gemma3")
input_text = st.text_input("What question do you have in mind?")

# Ollama Gemma3 model
llm = Ollama(model="gemma3")
output_parser = StrOutputParser()
chain = prompt | llm | output_parser

if input_text:
    with st.spinner("Generating response..."):
        response = chain.invoke({"question": input_text})
        st.write(response)
