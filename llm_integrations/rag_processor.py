import os
import faiss
import numpy as np
import json
from sentence_transformers import SentenceTransformer

class RAGProcessor:
    """
    A class to handle document processing, FAISS index building, loading,
    and context retrieval for Retrieval Augmented Generation (RAG).
    """
    def __init__(self, document_dir: str, 
                 index_path: str = "data/faiss_index.idx", # Default index path relative to project root
                 embedding_model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initializes the RAGProcessor.

        Args:
            document_dir (str): Directory containing documents to process.
            index_path (str): Path to save/load the FAISS index.
            embedding_model_name (str): Name of the sentence transformer model to use.
        """
        self.document_dir = document_dir
        self.index_path = index_path
        # Define path for document chunks based on index_path
        self.chunks_path = os.path.splitext(self.index_path)[0] + "_chunks.json"
        self.embedding_model_name = embedding_model_name
        
        self.index = None
        self.document_chunks = []

        try:
            print(f"Loading embedding model: {self.embedding_model_name}...")
            self.embedding_model = SentenceTransformer(self.embedding_model_name)
            print("Embedding model loaded successfully.")
        except Exception as e:
            print(f"Error loading embedding model '{self.embedding_model_name}': {e}")
            self.embedding_model = None
            # Consider re-raising or specific handling if model loading is critical for all operations

        self.load_index() # Attempt to load an existing index

    def _load_and_chunk_documents(self) -> list[str]:
        """
        Loads documents from .txt and .md files in self.document_dir and splits them into chunks.
        Chunks by splitting on double newlines (paragraphs).
        """
        chunks = []
        if not os.path.exists(self.document_dir):
            print(f"Error: Document directory '{self.document_dir}' not found.")
            return chunks # Return empty list if directory doesn't exist

        print(f"Loading documents from: {self.document_dir}")
        for filename in os.listdir(self.document_dir):
            if filename.endswith((".txt", ".md")):
                filepath = os.path.join(self.document_dir, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        content = f.read()
                    # Simple chunking: split by paragraphs and filter empty strings
                    file_chunks = [chunk.strip() for chunk in content.split('\n\n') if chunk.strip()]
                    if file_chunks:
                        chunks.extend(file_chunks)
                        print(f"Loaded and chunked '{filename}'. Found {len(file_chunks)} chunks.")
                    else:
                        print(f"No non-empty chunks found in '{filename}'.")
                except Exception as e:
                    print(f"Error reading or chunking file {filepath}: {e}")
        
        if not chunks:
            print("No document chunks were loaded. Ensure your document directory is correct and files have content.")
        return chunks

    def build_index(self):
        """
        Builds a FAISS index from documents in self.document_dir.
        Saves the index and the corresponding document chunks.
        """
        if self.embedding_model is None:
            print("Error: Embedding model is not loaded. Cannot build index.")
            return

        print("Starting to build RAG index...")
        loaded_chunks = self._load_and_chunk_documents()

        if not loaded_chunks:
            print("No documents found or documents were empty. Index not built.")
            self.document_chunks = [] 
            self.index = None
            # Optionally remove old index files if they exist and no new chunks are found
            if os.path.exists(self.index_path): 
                try: os.remove(self.index_path) 
                except OSError as e: print(f"Error removing old index file {self.index_path}: {e}")
            if os.path.exists(self.chunks_path): 
                try: os.remove(self.chunks_path)
                except OSError as e: print(f"Error removing old chunks file {self.chunks_path}: {e}")
            return

        self.document_chunks = loaded_chunks 

        try:
            print(f"Generating embeddings for {len(self.document_chunks)} chunks...")
            embeddings = self.embedding_model.encode(
                self.document_chunks, 
                convert_to_tensor=False, 
                show_progress_bar=True
            )
            
            if not isinstance(embeddings, np.ndarray) or embeddings.ndim != 2:
                print(f"Error: Embeddings are not in the expected NumPy array format. Got shape: {embeddings.shape if isinstance(embeddings, np.ndarray) else type(embeddings)}")
                return

            print(f"Embeddings generated. Shape: {embeddings.shape}")

            dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatL2(dimension)
            self.index.add(embeddings.astype(np.float32)) 
            
            # Ensure data directory exists for index and chunks
            index_dir = os.path.dirname(self.index_path)
            if index_dir and not os.path.exists(index_dir): # Check if index_dir is not empty string
                os.makedirs(index_dir, exist_ok=True)
            
            chunks_dir = os.path.dirname(self.chunks_path)
            if chunks_dir and not os.path.exists(chunks_dir): # Check if chunks_dir is not empty string
                os.makedirs(chunks_dir, exist_ok=True)


            faiss.write_index(self.index, self.index_path)
            print(f"FAISS index saved to {self.index_path}")

            with open(self.chunks_path, 'w', encoding='utf-8') as f:
                json.dump(self.document_chunks, f, indent=4)
            print(f"Document chunks saved to {self.chunks_path}")
            
            print(f"Index built and saved successfully with {len(self.document_chunks)} chunks.")

        except Exception as e:
            print(f"Error building or saving index: {e}")
            self.index = None
            self.document_chunks = []

    def load_index(self):
        """
        Loads an existing FAISS index and corresponding document chunks from disk.
        """
        if not os.path.exists(self.index_path):
            print(f"Index file '{self.index_path}' not found. Cannot load index.")
            self.index = None
            self.document_chunks = []
            return
            
        if not os.path.exists(self.chunks_path):
            print(f"Chunks file '{self.chunks_path}' not found. Cannot load chunks. Index will not be loaded.")
            self.index = None # Invalidate index if chunks are missing
            self.document_chunks = []
            return

        try:
            print(f"Loading FAISS index from {self.index_path}...")
            self.index = faiss.read_index(self.index_path)
            print("FAISS index loaded.")

            print(f"Loading document chunks from {self.chunks_path}...")
            with open(self.chunks_path, 'r', encoding='utf-8') as f:
                self.document_chunks = json.load(f)
            print(f"Document chunks loaded ({len(self.document_chunks)} chunks).")
            
            if self.index.ntotal != len(self.document_chunks):
                print(f"Warning: Mismatch between FAISS index size ({self.index.ntotal}) and number of loaded chunks ({len(self.document_chunks)}). The index may be stale. Consider rebuilding.")
                # Depending on severity, you might want to invalidate the index here:
                # self.index = None
                # self.document_chunks = []
                # print("Index and chunks invalidated due to mismatch.")
            else:
                print("Index and document chunks loaded successfully and sizes match.")

        except Exception as e:
            print(f"Error loading index or chunks file: {e}")
            self.index = None
            self.document_chunks = []
            print("Index and chunks set to None due to loading error. Please rebuild the index if needed.")

    def retrieve_context(self, query: str, top_k: int = 3) -> str:
        """
        Retrieves relevant context for a given query from the FAISS index.

        Args:
            query (str): The user's query.
            top_k (int): The number of top relevant chunks to retrieve.

        Returns:
            str: A concatenated string of the retrieved context chunks, or an error/info message.
        """
        if self.embedding_model is None:
            return "Error: Embedding model is not available. Cannot retrieve context."
        if self.index is None or not self.document_chunks:
            return "RAG Index not available or empty. Please build or load it first."
        if self.index.ntotal == 0:
             return "RAG Index is empty. No context to retrieve."


        try:
            print(f"Retrieving context for query: '{query}' (top_k={top_k})")
            query_embedding = self.embedding_model.encode([query], convert_to_tensor=False)
            
            if not isinstance(query_embedding, np.ndarray) or query_embedding.ndim != 2:
                print(f"Error: Query embedding is not in the expected NumPy array format. Got shape: {query_embedding.shape if isinstance(query_embedding, np.ndarray) else type(query_embedding)}")
                return "Error: Failed to generate valid query embedding."

            actual_top_k = min(top_k, self.index.ntotal)
            if actual_top_k < top_k:
                print(f"Warning: Requested top_k ({top_k}) is greater than the number of items in the index ({self.index.ntotal}). Using top_k={actual_top_k}.")
            if actual_top_k == 0: # Should be caught by self.index.ntotal == 0 earlier
                return "No context available as the index is empty."

            distances, indices = self.index.search(query_embedding.astype(np.float32), actual_top_k)
            
            retrieved_chunks = []
            for i in indices[0]:
                if 0 <= i < len(self.document_chunks):
                    retrieved_chunks.append(self.document_chunks[i])
                else:
                    print(f"Warning: Invalid index {i} encountered during retrieval. Max index: {len(self.document_chunks)-1}. Skipping this chunk.")

            if not retrieved_chunks:
                return "No relevant context found for the query after filtering invalid indices (if any)."

            context_str = "\n\n---\n\n".join(retrieved_chunks) 
            print(f"Retrieved {len(retrieved_chunks)} chunks.")
            return context_str

        except IndexError as e: # Should be less likely with the check above
            print(f"Error: An index out of bounds occurred during context retrieval. Indices: {indices[0] if 'indices' in locals() else 'N/A'}. Chunks available: {len(self.document_chunks)}. Error: {e}")
            return "Error: Failed to retrieve context due to an indexing issue. Consider rebuilding the index."
        except Exception as e:
            print(f"Error during context retrieval: {e}")
            return f"Error during context retrieval: {e}"

if __name__ == '__main__':
    # Example Usage:
    sample_doc_dir = "data/sample_docs_for_rag" # Documents will be placed here
    index_file_path = "data/example_rag_index.idx" # Specific path for the example index

    # Ensure the sample document directory exists
    os.makedirs(sample_doc_dir, exist_ok=True)
    # Ensure the directory for the index file exists
    os.makedirs(os.path.dirname(index_file_path), exist_ok=True)


    with open(os.path.join(sample_doc_dir, "doc1.txt"), "w", encoding="utf-8") as f:
        f.write("The quick brown fox jumps over the lazy dog.\n\nThis is the first document about animals.\n\nFoxes are known for their cunning behavior.")
    
    with open(os.path.join(sample_doc_dir, "doc2.md"), "w", encoding="utf-8") as f:
        f.write("# Space Exploration\n\nThe final frontier, a vast expanse of stars and planets.\n\nExploring Mars is a key objective for NASA and other space agencies.")

    with open(os.path.join(sample_doc_dir, "doc3.txt"), "w", encoding="utf-8") as f:
        f.write("Artificial intelligence (AI) is rapidly evolving, with new breakthroughs happening frequently.\n\nMachine learning, a subset of AI, enables computer systems to learn from data without being explicitly programmed.")

    print("Created dummy documents for RAG testing in 'data/sample_docs_for_rag/'.")

    rag_processor = RAGProcessor(document_dir=sample_doc_dir, index_path=index_file_path)

    # Force rebuild for this example to ensure it runs through the process
    print("\nBuilding RAG index for sample documents (forcing rebuild for example)...")
    rag_processor.build_index()
    
    # If building failed, rag_processor.index would be None
    if rag_processor.index is not None and rag_processor.document_chunks:
        print("\nIndex built. Testing context retrieval...")
        
        queries = [
            "Tell me about foxes.",
            "What is NASA doing regarding Mars?",
            "Explain Artificial Intelligence.",
            "What is the color of the sky?" # Example of a query with potentially low relevance
        ]
        
        for q_idx, query in enumerate(queries):
            context = rag_processor.retrieve_context(query, top_k=2)
            print(f"\nQuery {q_idx+1}: {query}\nRetrieved Context:\n{context}")
            print("-" * 30)
    else:
        print("\nSkipping retrieval test as RAG index is not available for sample documents (build might have failed).")

    # Example of loading an existing index (if you run the script again without rebuilding)
    print("\nAttempting to load the index again (simulating a new session)...")
    rag_processor_load_test = RAGProcessor(document_dir=sample_doc_dir, index_path=index_file_path)
    if rag_processor_load_test.index is not None and rag_processor_load_test.document_chunks:
        print("Index loaded successfully on second instantiation.")
        context_after_load = rag_processor_load_test.retrieve_context("Tell me about foxes.", top_k=1)
        print(f"\nContext for 'foxes' after loading:\n{context_after_load}")
    else:
        print("Failed to load the index on second instantiation, or it was empty.")

    # Cleanup (optional: remove the created sample files and directories)
    # print("\nTo cleanup, manually delete the 'data/sample_docs_for_rag' directory and 'data/example_rag_index.idx', 'data/example_rag_index_chunks.json'.")The `llm_integrations/rag_processor.py` file has been written with the `RAGProcessor` class. I have reviewed the generated code against the detailed requirements.

**Verification of `RAGProcessor` Implementation:**

1.  **`__init__`**:
    *   Stores `document_dir`, `index_path`, `embedding_model_name`. (Correct)
    *   Constructs `chunks_path` correctly (e.g., `data/faiss_index_chunks.json` from `data/faiss_index.idx`). (Correct, using `os.path.splitext`)
    *   Initializes `self.index = None`, `self.document_chunks = []`. (Correct)
    *   Initializes `SentenceTransformer` and handles potential errors. (Correct)
    *   Calls `self.load_index()`. (Correct)

2.  **`_load_and_chunk_documents`**:
    *   Reads `.txt` and `.md` files from `self.document_dir`. (Correct)
    *   Handles `document_dir` not existing. (Correct)
    *   Splits content by `\n\n` and filters empty chunks. (Correct)
    *   Returns a list of text chunks. (Correct)
    *   Includes error handling for file operations and prints status messages. (Correct)

3.  **`build_index`**:
    *   Calls `_load_and_chunk_documents`. (Correct)
    *   Handles no chunks found (prints message, ensures index/chunks are empty, optionally removes old files). (Correct)
    *   Stores fresh chunks in `self.document_chunks`. (Correct)
    *   Generates embeddings using `self.embedding_model.encode` with correct parameters. (Correct)
    *   Handles potential non-NumPy array output from `encode`. (Correct)
    *   Creates `faiss.IndexFlatL2` and adds embeddings (with `astype(np.float32)`). (Correct)
    *   Ensures target directories for index and chunks files exist before writing. (Correct)
    *   Saves index using `faiss.write_index` and chunks using `json.dump`. (Correct)
    *   Prints status messages and handles errors. (Correct)

4.  **`load_index`**:
    *   Checks for existence of `self.index_path` and `self.chunks_path`. (Correct, with early returns if files are missing)
    *   Loads index using `faiss.read_index` and chunks using `json.load`. (Correct)
    *   Prints status messages. (Correct)
    *   Handles errors during loading, setting `self.index` and `self.document_chunks` to empty/None. (Correct)
    *   Includes a check for mismatch between index size and number of chunks. (Correct)

5.  **`retrieve_context`**:
    *   Checks if index/chunks are available and if embedding model is loaded. (Correct, also checks for `self.index.ntotal == 0`)
    *   Generates query embedding. (Correct)
    *   Handles potential non-NumPy array output from query encoding. (Correct)
    *   Searches index using `self.index.search`. (Correct)
    *   Adjusts `top_k` if it's larger than the number of items in the index. (Correct)
    *   Retrieves chunks from `self.document_chunks` using `indices[0]`. (Correct, with added bounds check for `i` before appending to `retrieved_chunks`)
    *   Concatenates chunks into a string. (Correct)
    *   Returns the context string or error messages. (Correct)
    *   Includes error handling for `IndexError` and other exceptions. (Correct)

6.  **Imports**: `os`, `faiss`, `numpy`, `json`, `SentenceTransformer` are all imported. (Correct)
7.  **Error Handling and Status Messages**: Basic try-except blocks and print statements are included as requested. (Correct)
8.  **`data/` directory**: Was created in a previous turn. The `__main__` block in `rag_processor.py` also ensures subdirectories within `data/` (like `data/sample_docs_for_rag/` and the directory for `data/example_rag_index.idx`) are created using `os.makedirs`. (Correct)
9.  **`__main__` block**: Provides a good example usage of the class, including creating dummy documents, building an index, and testing retrieval. This is helpful for testing and understanding the class. (Correct)

The code seems robust and covers all specified requirements. The addition of `os.path.splitext` for `chunks_path` is a clean way to derive it. The error handling and status messages are appropriate. The bounds check within `retrieve_context` before accessing `self.document_chunks[i]` is a good safeguard.
The example in `if __name__ == '__main__':` correctly uses `os.makedirs` for the `index_file_path`'s directory if it doesn't exist, which is good practice.
The default `index_path` in `__init__` is now `"data/faiss_index.idx"`, which makes sense for a project-relative path.
The logic for creating directories in `build_index` before saving files is also correctly implemented.
I am satisfied with the generated code.
