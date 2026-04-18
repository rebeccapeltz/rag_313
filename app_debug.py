import os
import sys
import io
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from sentence_transformers import CrossEncoder


# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL CONFIGURATION
# The local .env file contains an HF_TOKEN from huggingface.co. This prevents 
# some models from throwing an authentication error during download, 
# even if the token isn't strictly required for public models. 
# If you don't have an HF_TOKEN, you can create a free account on Hugging Face 
# and generate one in your settings.
#
# Suppress verbose logging from sentence_transformers and huggingface_hub,
# which are used by the cross-encoder reranker. This keeps the console output
# clean and focused on the assistant's responses rather than model loading info.

# contexlib.contextmanager is used to create a context manager that temporarily 
# redirects stdout and stderr to suppress output during the cross-encoder initialization, 
# which can be noisy. Use 'with' to wrap the block of code where you want to suppress output.
# ─────────────────────────────────────────────────────────────────────────────
from dotenv import load_dotenv
load_dotenv()

import logging
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

import contextlib

@contextlib.contextmanager
def suppress_output():
    with open(os.devnull, 'w') as devnull:
        old_stdout_fd = os.dup(1)
        old_stderr_fd = os.dup(2)
        os.dup2(devnull.fileno(), 1)
        os.dup2(devnull.fileno(), 2)
        try:
            yield
        finally:
            os.dup2(old_stdout_fd, 1)
            os.dup2(old_stderr_fd, 2)
            os.close(old_stdout_fd)
            os.close(old_stderr_fd)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1: Environment & Docker Model Runner (DMR) endpoint
#
# KMP_DUPLICATE_LIB_OK prevents a crash on macOS ARM when multiple copies of
# the OpenMP runtime get loaded (common with numpy + faiss on Apple Silicon).
#
# DMR_URL points to your local Docker Model Runner, which hosts both the
# embedding model and the LLM — no internet or API key required.
# ─────────────────────────────────────────────────────────────────────────────
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

DMR_URL = "http://localhost:12434/v1"
API_KEY = "not-needed"


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: Embedding model
#
# Embeddings convert text into a list of numbers (a "vector") that captures
# semantic meaning. The same model MUST be used here as in ingest.py —
# if they don't match, similarity scores will be meaningless.
#
# "ai/embeddinggemma" is the embedding model running in your Docker container.
# ─────────────────────────────────────────────────────────────────────────────
embeddings = OpenAIEmbeddings(
    model="ai/embeddinggemma",
    base_url=DMR_URL,
    api_key=API_KEY
)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: FAISS vector store
#
# FAISS (Facebook AI Similarity Search) is an index that stores all your
# document chunks as vectors on disk. Loading it here restores that index
# so we can search it at query time.
#
# allow_dangerous_deserialization=True is required by LangChain for FAISS
# because it uses pickle under the hood — only use indexes you created yourself.
# ─────────────────────────────────────────────────────────────────────────────
vectorstore = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)

# --- DEBUG: inspect what's actually stored ---
# docs = vectorstore.similarity_search("Standard Work Hours", k=4)
# print(f"Retrieved {len(docs)} docs")
# for i, doc in enumerate(docs):
#     print(f"\n--- DOC {i+1} ---")
#     print(repr(doc.page_content))  # repr shows hidden characters/whitespace
#     print("Metadata:", doc.metadata)

# --- DEBUG: dump ALL chunks in the index ---
# all_docs = vectorstore.docstore._dict
# print(f"Total chunks in index: {len(all_docs)}")

# from collections import Counter
# sources = [doc.metadata.get("source", "unknown") for doc in all_docs.values()]
# counts = Counter(sources)
# print("\nChunks per source:")
# for source, count in counts.items():
#     print(f"  {count} chunks | {source}")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4: LLM (Large Language Model)
#
# This is Llama 3.2 running locally via Docker Model Runner.
# temperature=0 means deterministic output — the model always picks the
# highest-probability token, which is best for factual Q&A.
# ─────────────────────────────────────────────────────────────────────────────
llm = ChatOpenAI(
    model="ai/llama3.2",
    base_url=DMR_URL,
    api_key=API_KEY,
    temperature=0
)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5: Retriever (first-stage, broad candidate fetch)
#
# The retriever does a fast vector similarity search: it converts your query
# to a vector and finds the nearest neighbors in FAISS.
#
# WHY THIS CAN BE UNRELIABLE:
#   Vector similarity captures "topical closeness" but not precise relevance.
#   For example, a chunk about "vacation leave" might score similar to a query
#   about "sick leave" just because both are about leave policies.
#
# STRATEGY: Fetch MORE candidates (k=20) than you ultimately need, then let
# the reranker (Section 6) pick the best ones. This is called "retrieve & rerank."
# Note that increasing k improves recall (more chances to get the right chunk) but also
# increases latency, so it's a tradeoff.
# ─────────────────────────────────────────────────────────────────────────────

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 20}
)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6: Cross-Encoder Reranker (second-stage, precision scoring)
#
# A cross-encoder reads the query AND a chunk together as a pair, then outputs
# a relevance score. Unlike embeddings (which encode query and chunk separately),
# the cross-encoder sees both at once — this makes it far more accurate.
#
# "cross-encoder/ms-marco-MiniLM-L-6-v2" is a lightweight model (~80MB)
# trained specifically for passage relevance on MS MARCO (a QA dataset).
# It runs on CPU and is fast enough for interactive use.
#
# HOW IT WORKS:
#   1. We get 20 candidate chunks from FAISS.
#   2. We score each (query, chunk) pair using the cross-encoder.
#   3. We sort by score descending and keep the top RERANK_TOP_K chunks.
#   4. Only those top chunks go to the LLM as context.
#
# FIRST RUN: The model (~80MB) will be downloaded automatically from
# Hugging Face and cached locally (~/.cache/huggingface/).
#
# We are suppressing the stdout during initialization because the cross-encoder
# can be verbose about loading the model, and we want to keep the console output
# clean for the assistant's responses. The suppress_output context manager
# temporarily redirects stdout and stderr to null while the model loads, then restores it.
# ─────────────────────────────────────────────────────────────────────────────

# Suppress the CrossEncoder stdout load report
try:
    _orig_stdout, _orig_stderr = sys.stdout, sys.stderr
    sys.stdout = sys.stderr = io.StringIO()

    with suppress_output():
        reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
finally:
    sys.stdout, sys.stderr = _orig_stdout, _orig_stderr

RERANK_TOP_K = 4  # how many chunks to pass to the LLM after reranking


def rerank_docs(query: str, docs: list) -> list:
    """
    Score each (query, doc) pair with the cross-encoder and return
    the top RERANK_TOP_K docs sorted by relevance score.
    """
    if not docs:
        return docs

    pairs = [(query, doc.page_content) for doc in docs]
    scores = reranker.predict(pairs)                  # returns a list of floats
    scored = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)


    # --- DEBUG: print reranker scores ---
    # print("\n--- Reranker scores ---")
    # for score, doc in scored:
    #     source = doc.metadata.get("source", "unknown")
    #     print(f"  {score:.4f} | {source} | {doc.page_content[:60]}")

    return [doc for _, doc in scored[:RERANK_TOP_K]]


def format_docs(docs: list) -> str:
    """Join document chunks into a single context string for the prompt."""
    return "\n\n".join(doc.page_content for doc in docs)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7: RAG prompt
#
# This template structures what the LLM receives. The {context} slot gets
# filled with the reranked document chunks; {question} gets the user's query.
#
# The instruction "If the context does not contain the answer..." is a safety
# rail that reduces hallucination — it tells the LLM to admit uncertainty
# rather than invent an answer.
# ─────────────────────────────────────────────────────────────────────────────
template = """
You are a helpful assistant. Use the context below to answer the question.
If the context does not contain the answer, say: "I could not find that information."

Context:
{context}

Question: {question}

Answer:
"""

prompt = ChatPromptTemplate.from_template(template)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8: RAG chain (retrieve → rerank → generate)
#
# LangChain's LCEL (LangChain Expression Language) chains steps with | pipes.
# Each step's output becomes the next step's input.
#
# FLOW:
#   user query
#     → retriever        (FAISS: query → top-10 candidate chunks)
#     → rerank_lambda    (cross-encoder: top-10 → top-4 by relevance)
#     → format_docs      (join chunks into one context string)
#     → prompt           (fill template with context + question)
#     → llm              (Llama 3.2 generates the answer)
#     → StrOutputParser  (extract the plain text string from the LLM response)
#
# RunnableLambda wraps a plain Python function so it can participate
# in the LCEL pipeline. The lambda captures the query for the reranker
# because at this point in the chain, the query is still available via
# RunnablePassthrough in the parallel dict step above.
# ─────────────────────────────────────────────────────────────────────────────
def make_rerank_lambda(query_ref: dict):
    """
    Returns a RunnableLambda that closes over the query string so the
    reranker can score (query, chunk) pairs.
    """
    def _rerank(docs):
        return rerank_docs(query_ref["question"], docs)
    return RunnableLambda(_rerank)


# We use a two-step approach: capture query in a passthrough dict,
# then pipe into the reranker before formatting.
rag_chain = (
    RunnablePassthrough.assign(
        # Retrieve candidates, rerank them, format as a single string
        context=RunnableLambda(
            lambda inputs: format_docs(
                rerank_docs(
                    inputs["question"],
                    retriever.invoke(inputs["question"])
                )
            )
        )
    )
    | prompt
    | llm
    | StrOutputParser()
)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 9: User interface (CLI loop)
#
# A simple read-eval-print loop. The chain is invoked with a dict because
# the prompt template expects both "question" and "context" keys — the chain
# fills "context" internally via the retriever + reranker pipeline.
# ─────────────────────────────────────────────────────────────────────────────
print("\n")
print("Welcome to the HR Assistant!\n")
print("Ask about HR policies, benefits, or procedures.\n")
print("-- Type 'end' to exit the assistant. --\n")
print("Sample query: 'What can I use sick leave for?'")

while True:
    user_query = input("\nEnter your question (or 'end' to quit): ")
    if user_query.lower() == "end":
        print("Goodbye!")
        break
    answer = rag_chain.invoke({"question": user_query})
    print("Answer:", answer)
