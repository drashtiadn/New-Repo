from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
api_wrapper=WikipediaAPIWrapper(top_k_results=1,doc_content_chars_max=200)
wiki=WikipediaQueryRun(api_wrapper=api_wrapper)

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings

loader = PyPDFLoader("Knowledgebase.pdf")
docs=loader.load()
documents=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200).split_documents(docs)
vectordb=Chroma.from_documents(documents,GoogleGenerativeAIEmbeddings(model="models/embedding-001"))
retriever=vectordb.as_retriever()

from langchain.tools.retriever import create_retriever_tool
retriever_tool=create_retriever_tool(retriever,"langsmith_search","Search for information about LangSmith. For any questions about LangSmith, you must use this tool!")

## DuckDuckGo Tool
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_community.tools import DuckDuckGoSearchResults

duckduckgo_wrapper = DuckDuckGoSearchAPIWrapper(region="de-de", time="d", max_results=2)
duckduckgo = DuckDuckGoSearchResults(api_wrapper=duckduckgo_wrapper, source="news")

tools = [retriever_tool, wiki, duckduckgo]

from dotenv import load_dotenv
load_dotenv()

import os
os.getenv("GOOGLE_API_KEY")
import google.generativeai as genai
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

from langchain_google_genai import ChatGoogleGenerativeAI
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature = 0)

from langchain import hub
# Get the prompt to use - you can modify this!
prompt = hub.pull("hwchase17/openai-functions-agent")

### Agents
from langchain.agents import create_openai_tools_agent
agent=create_openai_tools_agent(llm,tools,prompt)

## Agent Executer
from langchain.agents import AgentExecutor
agent_executor=AgentExecutor(agent=agent,tools=tools,verbose=True)

# Invoke the agent executor with user queries
response = agent_executor.invoke({"input": "Tell me about Langsmith"})
print(response)

# response = agent_executor.invoke({"input": "What's the paper 1605.08386 about?"})
# print(response)


# # Streamlit UI
# import streamlit as st
# st.title("RAG Q&A App")

# query = st.text_input("Ask a question:", "What is Langchain?")

# if st.button("Submit"):
#     response = agent_executor.invoke({"input": query})
#     st.write("Response:", response)        