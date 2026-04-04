import os
from langchain_community.vectorstores import FAISS

from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# -------------------------
# Path Setup
# -------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Project root: /local_rag
PROJECT_ROOT = os.path.dirname(BASE_DIR)

# Correct paths
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")

print("Running ingest from:", os.path.abspath(__file__))
print("Working directory:", os.getcwd())
print("Processed dir:", PROCESSED_DIR)

# -------------------------
# Load markdown files
# -------------------------

docs = []
for filename in os.listdir(PROCESSED_DIR):
    if filename.endswith(".md"):
        with open(os.path.join(PROCESSED_DIR, filename), "r") as f:
            text = f.read()
            docs.append(Document(page_content=text, metadata={"source": filename}))

print("Loaded docs:", len(docs))

if len(docs) == 0:
    raise ValueError("No markdown files found in processed directory!")

# -------------------------
# Chunk documents
# -------------------------

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(docs)

print("Chunks created:", len(chunks))

# -------------------------
# Embeddings
# -------------------------

embeddings = OpenAIEmbeddings(
    model="ai/embeddinggemma",
    base_url="http://localhost:12434/v1",
    api_key="not-needed"
)
test_vec = embeddings.embed_query("test")
print("Embedding dimension:", len(test_vec))

print(">>> ABOUT TO CALL from_documents()")

vectorstore = FAISS.from_documents(
    documents=chunks,
    embedding=embeddings
)
# vectorstore.save_local("faiss_index")
vectorstore.save_local(PROJECT_ROOT + "/faiss_index")

print(">>> FINISHED CALLING from_documents()")
print("Ingest complete!")

test_vec = embeddings.embed_query("hello world")
print("INGEST embedding dimension:", len(test_vec))
