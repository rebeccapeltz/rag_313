from langchain_community.vectorstores import FAISS as AFAISS
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings (
    model="ai/embeddinggemma",
    base_url="http://localhost:12434/v1",
    api_key="not-needed"
)

vectorstore = AFAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)
docs = vectorstore.similarity_search("test query", k=2)
print (f"Index loaded OK - {len(docs)} docs returned")


