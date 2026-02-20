import os
from dotenv import load_dotenv
from langchain_openai import OpenAI
from langchain_classic.agents import AgentType, initialize_agent, load_tools  # Legacy agents API moved to langchain_classic in LangChain 1.x
from langchain_community.callbacks import StreamlitCallbackHandler
import streamlit as st

load_dotenv()

llm = OpenAI(temperature=0, streaming=True, openai_api_key=os.getenv("OPENAI_API_KEY"))
tools = load_tools(["ddg-search"], llm=llm)
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    # verbose=True
)

# try: "what are the names of the kids of the 44th president of america"
# try: "top 3 largest shareholders of nvidia"
if prompt := st.chat_input():
    st.chat_message("user").write(prompt)
    with st.chat_message("assistant"):
        st.write("🧠 thinking...")
        st_callback = StreamlitCallbackHandler(st.container())
        response = agent.invoke({"input": prompt}, callbacks=[st_callback])
        st.write(response["output"])
