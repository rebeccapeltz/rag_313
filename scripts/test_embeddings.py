from langchain_openai import OpenAIEmbeddings

# -------------------------
# Embeddings
# -------------------------
#
# The embed_query() test call verifies the model is reachable and confirms the
# output dimension before committing to the full embedding job.

embeddings = OpenAIEmbeddings(
    model="ai/embeddinggemma",
    base_url="http://localhost:12434/v1",   # Local Docker Model Runner endpoint
    api_key="not-needed"                    # No authentication required for local inference
)

# Test to see the embdding dimension of the word "test"
test_vec = embeddings.embed_query("test")
print("Embedding dimension:", len(test_vec))