"""
Goal: Build an app where you can upload a PDF (e.g., a research paper or manual) and ask questions about its content.
Concepts: Document loaders, text splitting, vector stores (like FAISS), retrieval-augmented generation (RAG).
"""

import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain_community.llms import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

st.set_page_config(page_title="Chat with your PDF")
st.title("Chat with your PDF")

query = st.text_input("Ask a question about the PDF")

if query:
    embeddings = OpenAIEmbeddings(model=os.getenv('TEXT_EMBEDDING_LARGE_3'))
    db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = db.similarity_search(query)

    llm = OpenAI(temperature=0)
    chain = load_qa_chain(llm, chain_type="stuff")
    response = chain.run(input_documents=docs, question=query)

    st.write(response)