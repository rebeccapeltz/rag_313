# Local RAG App (FAISS + LangChain + Docker Model Runner)

This project is a lightweight Retrieval-Augmented Generation (RAG) system that is useful for
learning how RAG works.  Claude.AI helped me with coding and documenting the app.py script.  The models used can be run locally using Docker Model Runner.  At least 16 GB of memory is needed to run the models and the application script and 32 GB is the recommended amount of memory.

It runs entirely locally using:

- FAISS for vector search
- LangChain for chaining
- Docker Model Runner (DMR) for both embeddings and LLM inference models
- Four local HR policy documents in PDF format as the knowledge base

The app loads a FAISS index, retrieves relevant chunks, and answers user questions 
using a local LLM.  The FAISS index cannot be used with version of Python greater than 3.13.

The documents have been converted from PDF to Markdown text and loaded into a FAISS index.  If you want to change the documents or try out the conversion and ingestion scripts, you'll find the ata in the documents in the /data directory and the scripts for conversion and extraction in the /scripts directory.

## Requirements
This project requires Python 3.13. The FAISS library (faiss-cpu), used for vector similarity search, does not currently provide pre-built packages for Python 3.14. Attempting to run this project on Python 3.14 will likely result in installation errors."

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
Download and Install `Docker Desktop`: 
  - [MAC](https://docs.docker.com/desktop/setup/install/mac-install/)
  - [LINUX](https://docs.docker.com/desktop/setup/install/linux/)
  - [WINDOWS](https://docs.docker.com/desktop/setup/install/windows-install/)  

### Enable Docker Model Runner with TCP
You can use the  **Settings** in the Docker User Interface to enable AI with TCP.   You need TCP to enable local applications to communicate with the AI models.

#### Docker User Interface
Click on `Settings` (Gear Icon) in the Docker User Interface and then `AI`.  Check box to enable AI and TCP. Enablings TCP allows you to interact with tools like VS Code.  Click on `Close` in the lower right hand corner to leave settings and return to the main menu.

![Settings turn on AI and TCP](./images/docker-enable-ai.png). 

The Docker Desktop application has an easy to use interface, but it's a good idea to run `pull` command to control Docker Models from your computer's command line. The Docker CLI supports MAC, Windows and Linux, but not all the commands work on all platforms.  The commands listed below should work on all platforms.

#### Pull and List Models 

After you have successfully "pulled" the models from Docker, you will see them listed by clicking on the **Models** tab on the left.

```bash

# Pull models
# Model used to create FAISS Index/Ingestion Pipeline
docker model pull ai/embeddinggemma

# LLM used for Query Pipeline
docker model pull ai/llama3.2

# List Downloaded Models
docker model list

# Run the Chat Model 
docker model run ai/llama3.2

# End Chat Model Conversation
/bye

# Run the Embeddings Model
# curl will make an API call to convert text to a numeric vector
curl --location 'http://localhost:12434/engines/llama.cpp/v1/embeddings' \
--header 'Content-Type: application/json' \
--data '{ "model": "ai/embeddinggemma", "input": "Your text to embed here" }'

```


Two models are used to used for this RAG projects: one as the LLM to which prompts are submitted, and one to hold embeddings. The embeddings are stored in FAISS index which is like a local database.  

The RAG project requires two models be pulled:
1. `ai/llama3.2` serves as LLM Inference provider so that you can ask AI questions.
2. `ai/embeddinggemma` provides a method to create text embeddings. These embeddings are numerical representations. Part of the process of setting up RAG is adding content and making that content retrievable py the Inference model.

If you're running locally you want to choose model that don't requires a lot of parameters in order to save manage resources. Both of the models suggested above will be available and efficent.

![Docker Model Runnner with LLM and Embeddings running](./images/list-model-ui-docker.png)

To learn more about using Docker Model Runner to host AI models on your local machine, see this <a href="https://medium.com/@code-literacy/docker-model-runner-wow-5397090b3251" target="_blank">Docker Model Runner Blog Post</a>.

### Reranking
Reranking is part of the RAG Retrieval Pipeline. It compares text from the prompt to retrieved data.  thr cross-encoder reads the query text and the data together and outputs a rank score. The top scores will provide the context for the LLM.

Importing the `sentence_transformer` packages provides the `cross-encoder/ms-marco-MiniLM-L-6-v2`.  This mini model is downloaded and used for ranking FAISS index content captured in the embeddings model.  The ranked FAISS index data is sorted and the top 4 chunks are passed to the `llama3.2` LLM as context.  The `cross-encoder/ms-marco-MiniLM-L-6-v2` will be downloaded and cached on your local drive under your user directory the first time you run the code.

## Logging
I've added logging adjustments to prevent warnings that aren't relevant to running the code. You should still get logging errors when there are errors, but not for warnings.  See the documentation in the code for this.

## Install Python Application and Run

1. Install Python version 3.13.
2. Create a virtual environment: `python3.13 -m venv .venv`.
3. Activate the virtual environment: `source .venv/bin/activate` (MAC).    
or `.venv\Scripts\activate` (WINDOWS COMMAND PROMPT). 
4. You can terminate the virtual environment with this command: `deactivate`.
5. Install packages: `pip install -r requirements.txt`. 
6. Set up Docker to load and run the two models: ai/llama3.2 and ai/embeddinggemma 
7. Implement the Data Ingestion Pipeline (below).
8. Run the app: `python app.py`. 
9. (Optional) If you're using this to learn how the RAG flow behaves, you can run the `app_debug.py` script to get information back at each step.  

Depending on the memory in your local hardware, the app may be slow to respond.

## Data Ingestion Pipeline

1. Raw data (.pdf's) are located in ./data/raw
2. Processed data (.md) is generated using ./scripts/convert.py
3. Data is loaded into embeddings using scripts/ingest.py which creates ./faiss_index
4. Prompts are created and serviced in ./app.py

### Conversion and Ingestion Pipeline
![Conversion and Ingestion Pipeline](./images/ingestion_pipeline.png)

### Query Pipeline
![Query Pipeline](./images/query_pipeline.png)



## Example: Human Resources Standard Operating Procedures

The sample content that will be accessible in this RAG will help to answer questions that users have about Human Resources.  Building on this could create a tool used by any employee to lookup information from Human Resources.

