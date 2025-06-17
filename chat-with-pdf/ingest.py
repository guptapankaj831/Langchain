from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from dotenv import load_dotenv
import os

load_dotenv()

def ingest_pdf(pdf_path: str, index_path: str = "faiss_index"):
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
    chunks = splitter.split_documents(docs)

    embedding = OpenAIEmbeddings(model=os.getenv('TEXT_EMBEDDING_LARGE_3'))
    db = FAISS.from_documents(chunks, embedding)
    db.save_local(index_path)
    print(f"Indexed {len(chunks)} chunks to {index_path}.")

if __name__ == "__main__":
    import sys
    ingest_pdf(sys.argv[1])
