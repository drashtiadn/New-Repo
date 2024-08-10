from fastapi import FastAPI, Request, UploadFile, File
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.document_loaders import PDFPlumberLoader
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import PromptTemplate
from langchain.tools.retriever import create_retriever_tool
from langchain_community.utilities import WikipediaAPIWrapper, DuckDuckGoSearchAPIWrapper
from langchain_community.tools import WikipediaQueryRun, DuckDuckGoSearchRun
from langchain.agents import create_openai_tools_agent, AgentExecutor
import os

app = FastAPI()

folder_path = "db"

# ---- Google Chat Model ----
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

embedding = FastEmbedEmbeddings()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1024, chunk_overlap=80, length_function=len
)


def get_response(user_query, chat_history):

    template = """
    You are a helpful assistant. Answer the following questions considering the history of the conversation:

    Chat history: {chat_history}

    User question: {user_question}
    """

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

    vector_store = Chroma(persist_directory=folder_path, embedding_function=embedding)

    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 20, "score_threshold": 0.1},
    )

    document_chain = create_stuff_documents_chain(cached_llm, raw_prompt)
    chain = create_retrieval_chain(retriever, document_chain)

    result = chain.invoke({"input": query})

    sources = [{"source": doc.metadata["source"], "page_content": doc.page_content} for doc in result["context"]]

    response_answer = {"answer": result["answer"], "sources": sources}
    return response_answer


@app.post("/pdf/")
async def pdfPost(file: UploadFile = File(...)):
    file_name = file.filename
    save_file = os.path.join("pdf", file_name)

    with open(save_file, "wb") as f:
        f.write(await file.read())

    loader = PDFPlumberLoader(save_file)
    docs = loader.load_and_split()

    chunks = text_splitter.split_documents(docs)

    vector_store = Chroma.from_documents(
        documents=chunks, embedding=embedding, persist_directory=folder_path
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

    vector_store = Chroma(persist_directory=folder_path, embedding_function=embedding)

    retriever_tool = create_retriever_tool(vector_store, "langsmith_search", "Search for information about LangSmith. For any questions about LangSmith. You should use this")

    api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200)
    wiki = WikipediaQueryRun(api_wrapper=api_wrapper)

    duckduckgo_wrapper = DuckDuckGoSearchAPIWrapper(top_k_results=1, doc_content_chars_max=200)
    duckduckgo = DuckDuckGoSearchRun(api_wrapper=duckduckgo_wrapper)

    tools = [retriever_tool, wiki, duckduckgo]

    agent = create_openai_tools_agent(llm, tools, raw_prompt)

    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    response = agent_executor.invoke({"input": query})

    response_answer = {"answer": response}
    return response_answer
