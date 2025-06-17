from dotenv import load_dotenv
import os
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader

def ingest_diary(folder_path="diary_entries", index_path="faiss_index"):
    all_docs = []

    for file in os.listdir(folder_path):
        if file.endswith('.txt'):
            path = os.path.join(folder_path, file)
            loader = TextLoader(path)
            all_docs.extend(loader.load())
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=40)
    chunks = splitter.split_documents(all_docs)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    db = FAISS.from_documents(chunks, embeddings)
    db.save_local(index_path)

if __name__ == "__main__":
    ingest_diary()
