# from langchain_community.tools import WikipediaQueryRun
# from langchain_community.utilities import WikipediaAPIWrapper


# api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200)
# wiki = WikipediaQueryRun(api_wrapper=api_wrapper)


# from langchain_community.document_loaders import WebBaseLoader
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from langchain_community.vectorstores import FAISS
# from langchain_text_splitters import RecursiveCharacterTextSplitter

# loader = WebBaseLoader("https://docs.smith.langchain.com/")
# docs = loader.load()
# documents = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(docs)

# # Provide the Google API key as an argument to GoogleGenerativeAIEmbeddings
# vectordb = FAISS.from_documents(documents, GoogleGenerativeAIEmbeddings(google_api_key="AIzaSyBK0XZOOwQNb6tse5t3-DK9Mi15WuWDe-I"))
# retriever = vectordb.as_retriever()

# from langchain.tools.retriever import create_retriever_tool
# retriever_tool = create_retriever_tool(retriever, "langsmith_search","Search for information about LangSmith.")

# from langchain_community.utilities import ArxivAPIWrapper
# from langchain_community.tools import ArxivQueryRun

# arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200)
# arxiv = ArxivQueryRun(api_wrapper=arxiv_wrapper)

# tools = [wiki, arxiv, retriever_tool]

# from dotenv import load_dotenv
# load_dotenv()

# # No need to import ChatOpenAI again, it was already imported earlier
# # from langchain_openai import ChatOpenAI
# from langchain_google_genai import ChatGoogleGenerativeAI


# # Create LLM agent
# llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

# from langchain import hub
# prompt = hub.pull("hwchase17/openai-functions-agent")

# from langchain.agents import create_openai_tools_agent
# agent = create_openai_tools_agent(llm, tools, prompt)

# from langchain.agents import AgentExecutor
# agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# # Invoke the agent executor with user queries
# response = agent_executor.invoke({"input": "Tell me about Langsmith"})
# print(response)

# response = agent_executor.invoke({"input": "What's the paper 1605.08386 about?"})
# print(response)


# # Streamlit UI
# import streamlit as st
# st.title("RAG Q&A App")

# query = st.text_input("Ask a question:", "What is Langchain?")

# if st.button("Submit"):
#     response = agent_executor.invoke({"input": query})
#     st.write("Response:", response)



from fastapi import FastAPI, Request, UploadFile, File
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PDFPlumberLoader
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import PromptTemplate
from langchain.tools.retriever import create_retriever_tool
from langchain_community.utilities import WikipediaAPIWrapper, DuckDuckGoSearchAPIWrapper
from langchain_community.tools import WikipediaQueryRun, DuckDuckGoSearchRun
from langchain.agents import create_openai_tools_agent, AgentExecutor
import streamlit as st
import os

app = FastAPI()

folder_path = "db"
pdf_folder = "pdf"

# Ensure directories exist
os.makedirs(folder_path, exist_ok=True)
os.makedirs(pdf_folder, exist_ok=True)

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1024, chunk_overlap=80, length_function=len
)

raw_prompt = PromptTemplate.from_template(
    """ 
    Answer question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n{context}?\n
    Question: \n{question}\n

    Answer:
    """
)

@app.post("/ai/")
async def aiPost(request: Request):
    json_content = await request.json()
    query = json_content.get("query")

    response = llm.invoke(query)

    response_answer = {"answer": response}
    return response_answer


@app.post("/ask_pdf/")
async def askPDFPost(request: Request):
    json_content = await request.json()
    query = json_content.get("query")

    vector_store = Chroma(persist_directory=folder_path, embedding_function=embeddings)

    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 20, "score_threshold": 0.1},
    )

    document_chain = create_stuff_documents_chain(llm, raw_prompt)
    chain = create_retrieval_chain(retriever, document_chain)

    result = chain.invoke({"input": query})

    sources = [{"source": doc.metadata["source"], "page_content": doc.page_content} for doc in result.get("context", [])]

    response_answer = {"answer": result["answer"], "sources": sources}
    return response_answer


@app.post("/pdf/")
async def pdfPost(file: UploadFile = File(...)):
    file_name = file.filename
    save_file = os.path.join(pdf_folder, file_name)

    with open(save_file, "wb") as f:
        f.write(await file.read())

    loader = PDFPlumberLoader(save_file)
    docs = loader.load_and_split()

    chunks = text_splitter.split_documents(docs)

    vector_store = Chroma.from_documents(
        documents=chunks, embedding_function=embeddings, persist_directory=folder_path
    )

    vector_store.persist()

    response = {
        "status": "Successfully Uploaded",
        "filename": file_name,
        "doc_len": len(docs),
        "chunks": len(chunks),
    }
    return response


@app.post("/agent_query/")
async def agentQueryPost(request: Request):
    json_content = await request.json()
    query = json_content.get("query")

    vector_store = Chroma(persist_directory=folder_path, embedding_function=embeddings)

    retriever_tool = create_retriever_tool(vector_store, "Search for information about Bus Information. For any questions about Bus Timing. You should use this")

    api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200)
    wiki = WikipediaQueryRun(api_wrapper=api_wrapper)

    duckduckgo_wrapper = DuckDuckGoSearchAPIWrapper(top_k_results=1, doc_content_chars_max=200)
    duckduckgo = DuckDuckGoSearchRun(api_wrapper=duckduckgo_wrapper)

    tools = [retriever_tool, wiki, duckduckgo]

    agent = create_openai_tools_agent(llm, tools, raw_prompt)

    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    response = agent_executor.run(query)  # Updated method call

    response_answer = {"answer": response}
    return response_answer

# The below methods need to be defined for Streamlit integration
def user_input(user_question):
    # Implement this method to process user input in Streamlit
    pass

def get_pdf_text(pdf_docs):
    # Implement this method to extract text from uploaded PDFs
    pass

def get_text_chunks(raw_text):
    # Implement this method to chunk text
    pass

def get_vector_store(text_chunks):
    # Implement this method to create and persist vector store
    pass

def main():
    st.set_page_config("Chat Bot")
    st.header("Intelligent Bus Inquiry Assistance Chat Bot 💁")

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
