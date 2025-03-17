# Caller Assistant for a College

The KIET College Caller Assistant is a conversational AI project that simulates a phone-based assistant for KIET College (including KIET Plus and KIET Womens). The assistant is designed to answer college-related queries in a formal and concise manner. This project demonstrates three different approaches to implement Retrieval Augmented Generation (RAG) and includes an embeddings pipeline for creating a custom dataset.

---
# Demo
[![Demo](https://via.placeholder.com/400x200.png?text=Click+to+Visit+Demo)](https://huggingface.co/spaces/rajavemula/custom-assistant-rag-ollama)

<img src="https://github.com/user-attachments/assets/e6b5274f-5c2a-46fb-91b3-f585f26df360" alt="Screenshot from 2025-03-17 18-32-40" width="600">


## Project Overview

The project uses a Streamlit web interface to simulate a natural phone conversation. It:
- **Retrieves relevant document chunks** based on user queries.
- **Translates queries** to improve clarity.
- **Constructs a conversation prompt** by combining retrieval context and chat history.
- **Streams AI-generated responses** either via a local model (Ollama) or an external API (Groq).

Additionally, the project includes an **embeddings.ipynb** Colab notebook that:
- **Loads a PDF** containing Q&A about the college (in JSON format).
- **Creates embeddings** using the `"all-MiniLM-L6-v2"` model, chosen for its CPU-friendly performance.
- **Tests retrieval** by loading the embeddings into an InMemoryVectorStore.
- **Saves the embeddings** to a JSON file for local use.

> **Note:** The dataset is synthetically generated and overall very limited. This approach serves as a proof-of-concept. Different data sources may require alternative models and retrieval techniques.

---

## Implementations

### 1. Basic RAG (`RAG(basic).py`)
- **Description:**  
  Implements a custom retrieval mechanism by encoding user queries and document chunks with the SentenceTransformer model. It calculates cosine similarity to retrieve relevant text snippets and applies keyword filtering. The final prompt is built and sent to a locally hosted model via the `ollama` command, with responses streamed character-by-character.
  
- **Key Features:**
  - Custom cosine similarity calculation.
  - Keyword-based filtering of document chunks.
  - Streaming response using a subprocess call to a local model via Ollama.

---

### 2. LangChain-based RAG (`RAG(lang_chain).py`)
- **Description:**  
  Leverages the LangChain library to simplify document management and retrieval via a FAISS vector store. It uses a custom LLM wrapper for local inference with Ollama and includes a query translation step to clarify ambiguous queries. A prompt template integrates conversation history and retrieved information for more context-aware responses.
  
- **Key Features:**
  - Utilizes FAISS and LangChain for efficient retrieval.
  - Custom LLM wrapper to interface with Ollama.
  - Query translation for accurate college references.
  - Enhanced prompt templating with conversation history.

---

### 3. API-based RAG (`RAG(via_api).py`)
- **Description:**  
  Similar to the LangChain-based approach, this implementation integrates with an external LLM API (Groq) for generating responses. It streams the model’s output token-by-token and uses environment variables (via dotenv) for configuration.
  
- **Key Features:**
  - Integrates with Groq API for LLM responses.
  - Streaming output for real-time responses.
  - Environment-driven configuration (e.g., API keys via a `.env` file).
  - Retains LangChain-based document retrieval and prompt construction.

---

## Embeddings Pipeline

The **embeddings.ipynb** Colab notebook includes scripts that:
- **Load Data:**  
  Extract Q&A data about the college from a PDF file (formatted in JSON).
- **Create Embeddings:**  
  Use the `"all-MiniLM-L6-v2"` model, chosen for its CPU-friendly performance, to generate embeddings for document chunks.
- **Test Retrieval:**  
  Load the generated embeddings into an InMemoryVectorStore to run test queries and verify retrieval performance.
- **Save Embeddings:**  
  Store the embeddings in a JSON file for local use in the RAG implementations.

> **Note:** The current dataset is synthetic and quite limited. The pipeline demonstrates a basic approach to data processing and retrieval. For larger or different datasets, consider using specialized models and customized retrieval techniques.

---

## Models & Local Testing with Ollama

For local testing, the project uses the [Ollama](https://ollama.com/) platform to run language models:
- **Installation:**  
  Download and install Ollama on your machine.
- **Pull Models:**  
  Choose and pull models based on your system hardware (e.g., `qwen2.5:3b`). Adjust the model via environment variables if necessary.
- **Running Locally:**  
  Ensure the Ollama server is running in your terminal. Once active, the scripts will automatically invoke the required model via the `ollama` command. You can either start the server manually or run the scripts directly if the server is already active.

---

## Prerequisites

- **Python:** 3.8 or higher  
- **Libraries:**
  - Streamlit
  - SentenceTransformer (for Basic RAG)
  - LangChain and langchain_community libraries
  - FAISS (faiss-cpu)
  - NumPy, JSON, subprocess, time, os, re  
  - python-dotenv (for API-based implementation)
  
- **Models & Tools:**
  - Local model integration using [Ollama](https://ollama.com/) for Basic and LangChain-based RAG.
  - Groq API credentials for the API-based approach.

---

## Installation

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/yourusername/your-repo-name.git
   
   cd your-repo-name

Create and Activate a Virtual Environment:

    python -m venv venv
   
    source venv/bin/activate  # On Windows: venv\Scripts\activate

Install Dependencies:
run:

    pip install -r requirements.txt

Set Up Ollama:
    Download and install Ollama from Ollama's website.
    Pull the required model based on your system hardware (e.g., qwen2.5:3b).
    Start the Ollama server in your terminal before running the RAG scripts.

Usage

Each implementation is designed to run as a Streamlit app. To start the assistant, run the corresponding file:

Basic RAG:

    streamlit run RAG(basic).py

LangChain-based RAG:

    streamlit run RAG(lang_chain).py

API-based RAG:

    streamlit run RAG(via_api).py

To run and test the embeddings pipeline, open the embeddings.ipynb notebook in Google Colab.

Environment Variables (API-based RAG)

For the API-based implementation, create a .env file in the project root with the following variables:

    GROQ_API_KEY=your_groq_api_key_here


Differences Between Implementations

Basic RAG:
    Minimalistic and straightforward.
    Custom retrieval with cosine similarity and keyword filtering.
    Uses a local model via Ollama for generating responses.
    
LangChain-based RAG:
    Leverages LangChain’s document management and retrieval features.
    Uses FAISS for vector storage.
    Implements a query translation step for clarity.
    Integrates with a custom LLM wrapper for local inference.
    
API-based RAG:
    Uses the LangChain framework for retrieval and prompt construction.
    Replaces the local model with an external API (Groq) for LLM responses.
    Supports streaming responses through the Groq API.
    Configuration managed via environment variables.
    

Contributing

Contributions are welcome! Please open an issue or submit a pull request with improvements, bug fixes, or new features.
License

This project is licensed under the MIT License. See the LICENSE file for details.
Contact

For questions or support, please reach out at rajavemula44@gmail.com.
