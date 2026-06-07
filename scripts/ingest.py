"""
ingest.py — RAG Pipeline: Document Ingestion and Vector Index Builder
======================================================================
This script is the first stage of a Retrieval-Augmented Generation (RAG) pipeline.
It reads preprocessed markdown files, splits them into overlapping text chunks,
generates vector embeddings using a local embedding model, and saves the resulting
FAISS vector index to disk for use by the retrieval stage (app.py).

Pipeline overview:
    1. Load markdown (.md) files from the /data/processed directory
    2. Split each document into chunks using a sliding-window text splitter
    3. Embed each chunk via a locally-hosted embedding model (Docker Model Runner)
    4. Store all embeddings in a FAISS vector index and save it to /faiss_index

Dependencies:
    - langchain-community    : FAISS vector store wrapper
    - langchain-openai       : OpenAI-compatible embeddings client (used with local model)
    - langchain-text-splitters: Recursive character-based text splitting
    - langchain-core         : Base Document class
    - FAISS                  : Facebook AI Similarity Search library (vector storage/retrieval)

Local embedding model:
    Model  : ai/embeddinggemma (served via Docker Model Runner)
    Endpoint: http://localhost:12434/v1  (OpenAI-compatible API)
    No API key required for local inference.

Usage:
    Run from any working directory; paths are resolved relative to this file's location.
    $ python ingest.py

Output:
    /faiss_index/  — FAISS index files (index.faiss + index.pkl) ready for similarity search
"""

import os
from langchain_community.vectorstores import FAISS

from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# -------------------------
# Path Setup
# -------------------------
# Paths are resolved relative to this script's location so the script
# can be run from any working directory without breaking file references.
#
# Expected project layout:
#   RAG_313/
#   ├── src/
#   │   └── ingest.py          ← this file (BASE_DIR)
#   ├── data/
#   │   └── processed/         ← markdown source files (PROCESSED_DIR)
#   └── faiss_index/           ← output: FAISS index files (created on first run)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Directory containing this script

# Project root: /RAG_313
PROJECT_ROOT = os.path.dirname(BASE_DIR)               # One level up from src/

# Source directory containing preprocessed markdown files
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")

print("Running ingest from:", os.path.abspath(__file__))
print("Working directory:", os.getcwd())
print("Processed dir:", PROCESSED_DIR)

# -------------------------
# Load markdown files
# -------------------------
# Reads every .md file in the processed directory into a list of LangChain
# Document objects. Each Document carries the raw text as page_content and
# the filename as metadata, which is preserved through chunking so retrieved
# chunks can be traced back to their source document.

docs = []
for filename in os.listdir(PROCESSED_DIR):
    if filename.endswith(".md"):
        with open(os.path.join(PROCESSED_DIR, filename), "r") as f:
            text = f.read()
            docs.append(Document(page_content=text, metadata={"source": filename}))

print("Loaded docs:", len(docs))

print("PROCESSED_DIR resolves to:", os.path.abspath(PROCESSED_DIR))
print("Directory exists:", os.path.exists(PROCESSED_DIR))
print("All files in directory:", os.listdir(PROCESSED_DIR))
print("Markdown files found:", [f for f in os.listdir(PROCESSED_DIR) if f.endswith(".md")])
print("Docs loaded:", len(docs))
for doc in docs:
    print(f"  → {doc.metadata['source']} ({len(doc.page_content)} chars)")

if len(docs) == 0:
    raise ValueError("No markdown files found in processed directory!")

# -------------------------
# Chunk documents
# -------------------------
# Large documents are split into smaller, overlapping chunks before embedding.
# This is necessary because embedding models have a token limit, and smaller
# chunks produce more focused, semantically meaningful vectors.
#
# RecursiveCharacterTextSplitter tries to split on natural boundaries
# (\n\n, \n, spaces) before falling back to hard character cuts.
#
# chunk_size=1000   : Maximum characters per chunk (~200–250 tokens)
# chunk_overlap=200 : Characters shared between adjacent chunks, preserving
#                     context at split boundaries so no sentence is cut mid-thought

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(docs)

print("Chunks created:", len(chunks))

# -------------------------
# Embeddings
# -------------------------
# Converts each text chunk into a high-dimensional numeric vector (embedding)
# that captures its semantic meaning. Chunks with similar meaning will have
# vectors that are close together in vector space, enabling similarity search.
#
# The model is served locally via Docker Model Runner on an OpenAI-compatible
# endpoint, so we use LangChain's OpenAIEmbeddings client pointed at localhost.
# The api_key value is a required parameter but is not validated by the local server.
#
# The embed_query() test call verifies the model is reachable and confirms the
# output dimension before committing to the full embedding job.

embeddings = OpenAIEmbeddings(
    model="ai/embeddinggemma",
    base_url="http://localhost:12434/v1",   # Local Docker Model Runner endpoint
    api_key="not-needed"                    # No authentication required for local inference
)

print(">>> ABOUT TO CALL from_documents()")

# Build the FAISS vector store from all embedded chunks.
# from_documents() calls embed_documents() on each chunk and stores the
# resulting vectors in an in-memory FAISS index structure.
#
# save_local() writes two files to the faiss_index directory:
#   index.faiss — the binary FAISS index (vectors + search structure)
#   index.pkl   — pickled metadata mapping vector IDs back to Document objects
#
# These files are loaded by app.py at query time to perform similarity search.
vectorstore = FAISS.from_documents(
    documents=chunks,
    embedding=embeddings
)
# vectorstore.save_local("faiss_index")
vectorstore.save_local(PROJECT_ROOT + "/faiss_index")  # Save relative to project root

print(">>> FINISHED CALLING from_documents()")
print("Ingest complete!")


