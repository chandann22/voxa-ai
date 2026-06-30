from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
import os, glob
from dotenv import load_dotenv

load_dotenv()

DOCS_PATH   = "../docs"
CHROMA_PATH = "./chroma_db"

# Use free HuggingFace embeddings (no extra API key needed)
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

vectorstore = None

def load_documents():
    docs = []
    # Load .txt files
    for path in glob.glob(f"{DOCS_PATH}/*.txt"):
        loader = TextLoader(path)
        docs.extend(loader.load())
    # Load .pdf files
    for path in glob.glob(f"{DOCS_PATH}/*.pdf"):
        loader = PyPDFLoader(path)
        docs.extend(loader.load())
    return docs

def build_vectorstore():
    global vectorstore
    print("Building RAG vector store...")
    docs     = load_documents()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks   = splitter.split_documents(docs)
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PATH
    )
    print(f"RAG ready — {len(chunks)} chunks indexed.")

def search_docs(query: str, k: int = 3) -> str:
    if vectorstore is None:
        return ""
    results = vectorstore.similarity_search(query, k=k)
    if not results:
        return ""
    context = "\n\n".join([r.page_content for r in results])
    return context