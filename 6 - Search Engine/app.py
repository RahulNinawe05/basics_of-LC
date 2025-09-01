import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper,WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun,DuckDuckGoSearchResults
from langchain.agents import initialize_agent,AgentType
from langchain.callbacks import StreamlitCallbackHandler
import os
from dotenv import load_dotenv

## Arixiv and wiki wrapper
arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200)
arxiv=ArxivQueryRun(api_wrapper=arxiv_wrapper)

wiki_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200)
wiki = WikipediaQueryRun(api_wrapper=wiki_wrapper)

search = DuckDuckGoSearchResults(name="search")

st.title("langchain chat with search")

"""
When you run an agent or chain in LangChain, 
the agent takes steps (thinking, calling tools, 
getting results).
Normally, you only see the final answer.
With StreamlitCallbackHandler, 
you can also see the intermediate steps live in the Streamlit UI.
"""

## sliderbar for setting
st.sidebar.title("Setting")
api_key = st.sidebar.text_input("Enter Your groq api key")

if "messages" not in st.session_state:
    st.session_state["messages"]=[
        {"role":"assisstant", "content":"Hi, I'm a Chatbot who can search the web. how can i help you"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt:=st.chat_input(placeholder="What is Machine Learning"):
    st.session_state.messages.append({"role":"user","content":prompt})
    st.chat_message("user").write(prompt)

    llm = ChatGroq(groq_api_key=api_key, model="Llama3-8b-8192",streaming=True)

    tools = [search, wiki, arxiv]

    search_agent=initialize_agent(tools,llm,agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,handling_parsing_errors=True)

    with st.chat_message("assistant"):
        st_cb=StreamlitCallbackHandler(st.container(),expand_new_thoughts=True)
        responce = search_agent.run(st.session_state.messages,callbacks=[st_cb])
        st.session_state.messages.append({'role':"assistant", "content":responce})
        st.write(responce)