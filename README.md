# Local RAG App (FAISS + LangChain + Docker Model Runner)

This project is a lightweight Retrieval-Augmented Generation (RAG) system that is useful for
learning how RAG works.

It runs entirely locally using:

- FAISS for vector search
- LangChain for chaining
- Docker Model Runner (DMR) for both embeddings and LLM inference models
- Four ocal HR policy documents in PDF format as the knowledge base

The app loads a FAISS index, retrieves relevant chunks, and answers user questions 
using a local LLM.  The FAISS index cannot be used with version of Python greater than 3.13.


## Requirements
- Python 3.13
- Python packages
    - langchain-core
    - langchain-community
    - langchain-openai
    - faiss-cpu
    - sentence_transformers
    - dotenv
- Docker Model Runner with:
    - ai/embeddinggemma
    - ai/llama3.2 (or your chosen LLM)


## Docker Model Runner
Two models are used one as the LLM to which prompts are submitted, and one to hold embeddings.  To learn more about using Docker Model Runner to host AI models on your local machine, see this <a href="https://medium.com/@code-literacy/docker-model-runner-wow-5397090b3251" target="_blank">Docker Model Runner Blog Post</a>.

The RAG requires two models:
1. The first model serves as LLM Inference provider so that you can ask AI questions.
2. The second model provides a method to create text embeddings. These embeddings are numerical representations. Part of the process of setting up RAG is adding content and making that content retrievable py the Inference model.

If you're running locally you want to choose model that don't requires a lot of parameters in order to save manage resources. Both of the models suggested below will be available and efficent.

- ai/llama3.2 use for LLM Inference
- ai/embeddinggemma use for embeddings

The sentence_transformer packages provides the cross-encoder/ms-marco-MiniLM-L-6-v2.  This mini model is downloaded and used for ranking FAISS index content captured in the embeddings model.  The ranked FAISS index data is sorted and the top 4 chunks are passed to the llama3.2 LLM as context.

## Install and Run

1. Install Python version 3.13.
2. Create a virtual environment: `python3.13 -m venv .venv`.
3. Activate the virtual environment: `source .venv/bin/activate` (MAC).    
or `.venv\Scripts\activate` (WINDOWS COMMAND PROMPT).  
4. Install packages: `pip install -r requirements.txt`. 
5. Set up Docker to load and run the two models: ai/llama3.2 and ai/embeddinggemma 
6. Run the app: `python app.py`. 
7. (Optional) If you're using this to learn how the RAG flow behaves, you can uncomment the DEBUG and print statements to learn more about that.  

Depending on the memory in your local hardware, the app may be slow to respond.

## Data Flow

1. Raw data (.pdf's) are located in ./data/raw
2. Processed data (.md) is generated using ./scripts/convert.py
3. Data is loaded into embeddings using scripts/ingest.py which creates ./faiss_index
4. Prompts are created and serviced in ./app.py

## Example: Human Resources Standard Operating Procedures

The sample content that will be accessible in this RAG will help to answer questions that users have about Human Resources.  Building on this could create a tool used by any employee to lookup information from Human Resources.

