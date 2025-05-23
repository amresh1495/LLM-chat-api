# LLM Chatbot (Gemma 3 & Llama 3.2)

## Overview

This project provides a Gradio-based chatbot application that allows users to interact with two powerful open-source language models: Google's Gemma 3 (specifically the 12B instruction-tuned version) and Meta's Llama 3.2 (the 3B instruction-tuned version). The application supports dynamic model loading and unloading to help manage computational resources.

## Features

*   **Chat with Google Gemma 3 (12B IT)**: Interact with the `google/gemma-3-12b-it` model.
*   **Chat with Meta Llama 3.2 (3B Instruct)**: Interact with the `meta-llama/Llama-3.2-3B-Instruct` model.
*   **Retrieval Augmented Generation (RAG) from local documents**: Augment LLM responses with context retrieved from your own documents.
*   **Model Selection**: Choose the desired LLM from a dropdown menu in the UI.
*   **Dynamic Model Management**: Models are loaded on demand when selected. If one model is loaded and the user chooses to load the other, the first model is automatically unloaded to conserve GPU memory.
*   **Resource Cleanup**: A dedicated button allows users to clear the chat and explicitly unload any active model, freeing GPU resources.
*   **User-Friendly Interface**: Built with Gradio for an intuitive web-based chat experience.

## Project Structure

```
.
├── app.py                      # Main Gradio application script
├── config.py                   # Configuration file (currently reserved for future use)
├── requirements.txt            # Python dependencies for the project
├── llm_integrations/
│   ├── __init__.py
│   ├── gemma_client.py         # Client for interacting with Gemma models
│   ├── llama_client.py         # Client for interacting with Llama models
│   └── rag_processor.py        # Manages document processing, indexing, and retrieval for RAG
├── data/
│   └── .gitkeep                # Directory for RAG documents and FAISS index
├── tests/
│   ├── __init__.py
│   ├── test_app.py             # Unit tests for app.py
│   ├── test_llm_clients.py     # Unit tests for Gemma and Llama clients
│   └── test_rag_processor.py   # Unit tests for RAGProcessor
└── README.md                   # This file
```

*   **`app.py`**: Contains the Gradio interface logic, model loading, chat handling, RAG controls, and resource management.
*   **`config.py`**: Intended for future configuration options (e.g., different model IDs, generation parameters). Currently not used.
*   **`requirements.txt`**: Lists all Python packages required to run the application.
*   **`llm_integrations/`**: This directory holds the modules responsible for interacting with the specific LLMs and RAG.
    *   `gemma_client.py`: Implements the `GemmaClient` class to load and query Gemma models.
    *   `llama_client.py`: Implements the `LlamaClient` class to load and query Llama models.
    *   `rag_processor.py`: Implements the `RAGProcessor` class for document chunking, embedding, FAISS indexing, and context retrieval.
*   **`data/`**: This directory is used for RAG functionality.
    *   Place your `.txt` and `.md` documents here to be indexed for RAG.
    *   The FAISS index file (e.g., `faiss_index.idx`) and document chunks (e.g., `faiss_index_chunks.json`) will also be stored here.
*   **`tests/`**: Contains unit tests for the application.
    *   `test_app.py`: Tests the core logic of the Gradio application (`app.py`).
    *   `test_llm_clients.py`: Tests the functionality of `GemmaClient` and `LlamaClient`.
    *   `test_rag_processor.py`: Tests the functionality of `RAGProcessor`.

## Setup Instructions

1.  **Clone the Repository** (if applicable)
    If you have downloaded this project as a set of files, ensure they are all in a single project directory. If it's a Git repository:
    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```

2.  **Create and Activate a Python Virtual Environment** (Recommended)
    ```bash
    python -m venv venv
    ```
    On macOS/Linux:
    ```bash
    source venv/bin/activate
    ```
    On Windows:
    ```bash
    venv\Scripts\activate
    ```

3.  **Install Dependencies**
    Ensure your virtual environment is activated, then run:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Important Note on Model Access (Hugging Face)**
    *   To use the Gemma 3 and Llama 3.2 models, you must agree to their respective terms of use on Hugging Face:
        *   Gemma: Visit the Hugging Face model page for Gemma (e.g., `google/gemma-3-12b-it`) and accept the terms.
        *   Llama: Visit the Hugging Face model page for Llama (e.g., `meta-llama/Llama-3.2-3B-Instruct`) and accept the terms.
    *   You may also need to be authenticated with Hugging Face to download these models. If you haven't done so already, run the following command in your terminal and follow the prompts:
        ```bash
        huggingface-cli login
        ```
        This will store your Hugging Face token, allowing the `transformers` library to download gated models.

## How to Run

1.  Ensure your virtual environment is activated and all dependencies are installed.
2.  Navigate to the project's root directory in your terminal.
3.  Run the application:
    ```bash
    python app.py
    ```
4.  The Gradio interface will typically open automatically in your web browser. If not, open your browser and go to `http://127.0.0.1:7860` (the default Gradio address).

## How to Use

The application interface is organized into two main tabs: "LLM and RAG Controls" and "Chat Interface".

**1. LLM and RAG Controls Tab:**

*   **LLM Selection and Loading**:
    1.  **Select an LLM Model**: Choose either "Gemma 3 (12B IT)" or "Llama 3.2 (3B Instruct)" from the dropdown.
    2.  **Load Selected LLM**: Click this button to load the chosen model. Status messages will appear in the "LLM Status" box. This might take time, especially on the first load if model weights need downloading.
*   **RAG (Retrieval Augmented Generation) Controls**:
    1.  **Prepare Documents**: Place your custom `.txt` or `.md` document files into the `data/` directory located in the project's root folder.
    2.  **Initialize RAG Processor**: Click this button. It prepares the RAG system by loading the embedding model and any existing pre-built FAISS index from the `data/` directory. Check the "RAG Status" box for confirmation.
    3.  **Build/Update RAG Index**: After adding new documents or modifying existing ones in the `data/` directory, click this button. This will process all documents in `data/`, create embeddings, and build/update the FAISS index. This step is crucial for new documents to be searchable.
    4.  **Enable RAG for Chatbot Responses**: Check this checkbox if you want the chatbot to use the RAG system. When enabled, the chatbot will first search your documents for relevant context before generating a response. If unchecked, it will rely solely on its pre-trained knowledge.

**2. Chat Interface Tab:**

1.  **Ensure Model is Loaded**: Before chatting, make sure an LLM is loaded (see LLM controls).
2.  **Enable RAG (Optional)**: If you want to use RAG, ensure it's initialized, an index is built, and the "Enable RAG" checkbox is checked on the "LLM and RAG Controls" tab.
3.  **Chat**: Type your message into the message textbox and either click "Send" or press Enter. The model's response will appear in the chat window. If RAG is enabled, the response may be informed by your documents.
4.  **Switch Models**: To switch LLMs, go back to the "LLM and RAG Controls" tab, select the new model, and click "Load Selected LLM". The previously active LLM will be unloaded automatically.
5.  **Clear and Unload**: To clear the chat history and unload any active LLM (freeing GPU memory), click the "Clear Chat & Unload Models" button available on the "Chat Interface" tab. The RAG index and processor will remain loaded as they are less resource-intensive.

## Running Tests

To run the unit tests for this project:

1.  Ensure your virtual environment is activated and dependencies (including any test-specific ones, though none are explicitly separate here) are installed.
2.  From the project's root directory, run:
    ```bash
    python -m unittest discover tests
    ```
    Or, for more verbose output:
    ```bash
    python -m unittest -v discover tests
    ```

## Configuration

The `config.py` file is currently reserved for future configuration options (such as setting different default model IDs or generation parameters). At present, all configurations are handled directly within `app.py` or the client classes.

## Dependencies

The main dependencies for this project are:

*   **transformers**: For accessing and using pre-trained models from Hugging Face.
*   **torch**: The PyTorch library, required by `transformers` for model operations.
*   **gradio**: For creating the web-based user interface.
*   **accelerate**: To assist with model loading and efficient hardware utilization (e.g., automatic device mapping).
*   **sentence-transformers**: For generating embeddings from text documents for RAG.
*   **faiss-cpu**: For creating and searching the FAISS index for RAG. (`faiss-gpu` could be used for GPU acceleration if available and preferred).

Refer to `requirements.txt` for a complete list of dependencies and their specific versions.
