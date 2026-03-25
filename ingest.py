from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os


def load_documents():
    txt_loader = DirectoryLoader(
        path="data/",
        glob="*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"}
    )
    pdf_loader = PyPDFDirectoryLoader("data/")
    txt_docs=txt_loader.load()
    pdf_docs=pdf_loader.load()
    documents = txt_docs + pdf_docs
    print(f"Loaded {len(txt_docs)} text files and{len(pdf_docs)} pdf files")
    print(f"total:{len(documents)} documents")
    return documents


def split_documents(documents):
    splitter= RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n","."," ","\n\n"]
    )
    chunks=splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks")
    return chunks



def get_embeddings():
    embeddings= HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return embeddings


def create_and_save_index(chunks,embeddings):
    print("creating FAISS index....")
    vector_store = FAISS.from_documents(chunks,embeddings)
    vector_store.save_local("faiss_index")
    print("FAISS saved to faiss_index/")

if __name__== "__main__":
    print("Starting ingestion..")
    documents=load_documents()
    chunks=split_documents(documents)
    embeddings=get_embeddings()
    create_and_save_index(chunks,embeddings)
    print("ingestion process complete FAISS is ready.")
