import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough,RunnableLambda

# Optional: avoid OpenMP crash on macOS ARM
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# 1. Docker Model Runner endpoint
DMR_URL = "http://localhost:12434/v1"
API_KEY = "not-needed"

# 2. Embeddings (must match ingest.py)
embeddings = OpenAIEmbeddings(
    model="ai/embeddinggemma",
    base_url=DMR_URL,
    api_key=API_KEY
)

# Debug: confirm embedding dimension
# test_vec = embeddings.embed_query("hello world")
# print("APP embedding dimension:", len(test_vec))

# 3. Load FAISS index
vectorstore = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)
# print("\n--- DEBUG: Testing raw retrieval ---")
# docs = vectorstore.similarity_search("What are the options for sick leave?", k=4)
# print("Retrieved docs:", len(docs))
# for i, d in enumerate(docs):
#     print(f"\n--- DOC {i+1} ---")
#     print(d.page_content[:300])
#
# print("\n--- DEBUG: Testing retriever ---")

# 4. LLM
llm = ChatOpenAI(
    model="ai/llama3.2",
    base_url=DMR_URL,
    api_key=API_KEY,
    temperature=0
)

# 5. Retrieval configuration

retriever = vectorstore.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 4, "score_threshold": 0.0}
)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# 6. RAG prompt

template = """
You are a helpful assistant. Use the context below to answer the question.
If the context does not contain the answer, say: "I could not find that information."

Context:
{context}

Question: {question}

Answer:
"""

prompt = ChatPromptTemplate.from_template(template)

# print("\n--- DEBUG: RAG retriever output ---")
# rag_docs = retriever.invoke("What are my options for sick leave?")
# print("RAG retrieved:", len(rag_docs))
# for d in rag_docs:
#     print(d.page_content[:200])


# 7. RAG chain
rag_chain = (
    {
        "context": retriever | RunnableLambda(format_docs),
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)

# 8. 
# Test query 1

# query = "What recognition do I get on my first year anniversary?"
# print("\nQUERY:", query)
# print("ANSWER:", rag_chain.invoke(query))

# Test query 2

# query = "What can I use sick leave for?"
# print("\nQUERY:", query)
# print("ANSWER:", rag_chain.invoke(query))
 
## User Interface 
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
    answer = rag_chain.invoke(user_query)
    print("Answer:", answer)
