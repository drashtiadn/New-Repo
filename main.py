import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import Chroma
from langchain_community.document_loaders import PDFPlumberLoader
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import PromptTemplate
from langchain.tools.retriever import create_retriever_tool
from langchain_community.utilities import WikipediaAPIWrapper, DuckDuckGoSearchAPIWrapper
from langchain_community.tools import WikipediaQueryRun, DuckDuckGoSearchRun
from langchain.agents import create_openai_tools_agent, AgentExecutor
import os
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

# Setup environment variables and messages
load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature = 0)

### Construct retriever ###
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)
vectorstore = Chroma.from_documents(documents=splits, embedding=GoogleGenerativeAIEmbeddings())
retriever = vectorstore.as_retriever()


# ### Contextualize question ###
# contextualize_q_system_prompt = """Given a chat history and the latest user question \
# which might reference context in the chat history, formulate a standalone question \
# which can be understood without the chat history. Do NOT answer the question, \
# just reformulate it if needed and otherwise return it as is."""
# contextualize_q_prompt = ChatPromptTemplate.from_messages(
#     [
#         ("system", contextualize_q_system_prompt),
#         MessagesPlaceholder("chat_history"),
#         ("human", "{input}"),
#     ]
# )
# history_aware_retriever = create_history_aware_retriever(
#     llm, retriever, contextualize_q_prompt
# )


# ### Answer question ###
# qa_system_prompt = """You are an assistant for question-answering tasks. \
# Use the following pieces of retrieved context to answer the question. \
# If you don't know the answer, just say that you don't know. \
# Use three sentences maximum and keep the answer concise.\

# {context}"""
# qa_prompt = ChatPromptTemplate.from_messages(
#     [
#         ("system", qa_system_prompt),
#         MessagesPlaceholder("chat_history"),
#         ("human", "{input}"),
#     ]
# )
# question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

# rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)


# ### Statefully manage chat history ###
# store = {}


# def get_session_history(session_id: str) -> BaseChatMessageHistory:
#     if session_id not in store:
#         store[session_id] = ChatMessageHistory()
#     return store[session_id]


# conversational_rag_chain = RunnableWithMessageHistory(
#     rag_chain,
#     get_session_history,
#     input_messages_key="input",
#     history_messages_key="chat_history",
#     output_messages_key="answer",
# )


def main():
    st.set_page_config("Chat Bot")
    st.header(" Intelligent Bus Inquiry Assistance Chat Bot 💁")

    user_question = st.text_input("Ask me anything about bus schedules, routes, fares, and more!")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu")
        pdf_docs = st.file_uploader("Upload your Knowledge Base and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")


if __name__ == "__main__":
    main()