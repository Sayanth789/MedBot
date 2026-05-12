import os 

from langchain_huggingface import HuggingFaceEmbeddings 
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader 
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_data(data):
    loader = DirectoryLoader(data,
                            glob="*.pdf",
                            loader_cls=PyPDFLoader)
    documents = loader.load()

    return documents

    
def text_split(extracted_data):
    text_splitter = RecursiveCharacterSplitter(chunk_size=500, chunk_overlap=20)
    text_chunks = text_splitter.split_documents(extracted_data)

    return text_chunks    

def download_hugingface_embeddings():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embeddings 


