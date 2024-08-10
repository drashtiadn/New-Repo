from fastapi import FastAPI, Request, UploadFile, File
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import PromptTemplate
from langchain.tools.retriever import create_retriever_tool
from langchain_community.utilities import WikipediaAPIWrapper, DuckDuckGoSearchAPIWrapper
from langchain_community.tools import WikipediaQueryRun, DuckDuckGoSearchRun
from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain import hub
import streamlit as st
import os

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature = 0)

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# text_splitter = RecursiveCharacterTextSplitter(
#     chunk_size=1024, chunk_overlap=80, length_function=len
# )


def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader= PdfReader(pdf)
        for page in pdf_reader.pages:
            text+= page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store=Chroma.from_texts(text_chunks,embedding=embeddings)
    vector_store.add_documents("chromadb_index")

def get_conversational_chain():

    prompt_template="""
    Answer question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n{context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)

    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    
    return chain


def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")

    new_db = Chroma.from_documents("chromadb_index", embeddings, allow_dangerous_deserialization=True,)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()


    response = chain(
        {"input_documents":docs, "question":user_question}
        , return_only_outputs=True)
    
    print(response)
    st.write("Reply: ", response["output_text"])




def main():
    st.set_page_config("Chat PDF")
    st.header("Chat with PDF using Gemini 💁")

    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")


if __name__ == "__main__":
    main()