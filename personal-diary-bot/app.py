import streamlit as st
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
import os
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

load_dotenv()

st.set_page_config(page_title="Personal Diary Q&A")
st.title("📘 Personal Diary Q&A Bot")

query = st.text_input("Ask something about your diary:")
if query:
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    db = FAISS.load_local('faiss_index', embeddings, allow_dangerous_deserialization=True)
    retriever = db.as_retriever(search_kwargs={"k": 10})

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff")
    answer = qa_chain.run(query)
    st.write("**Answer:**", answer)

