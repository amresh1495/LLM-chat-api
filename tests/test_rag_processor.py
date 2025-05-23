import unittest
from unittest.mock import patch, MagicMock, mock_open, call
import os
import json
import numpy as np

# Mock faiss before it's imported by RAGProcessor
# This is a common way to ensure the mock is in place when the module under test loads it.
sys_modules_patch = patch.dict('sys.modules', {'faiss': MagicMock()})
sys_modules_patch.start()
import faiss # Now faiss is the MagicMock

try:
    from llm_integrations.rag_processor import RAGProcessor
except ImportError:
    RAGProcessor = None # Placeholder if import fails

# Stop the sys.modules patch after RAGProcessor is imported so faiss is back to normal for other tests if any
sys_modules_patch.stop()


class TestRAGProcessor(unittest.TestCase):

    # Patch SentenceTransformer for all tests in this class
    @patch('llm_integrations.rag_processor.SentenceTransformer')
    def setUp(self, MockSentenceTransformer):
        if RAGProcessor is None:
            self.skipTest("RAGProcessor class not imported.")

        # Configure the mock for SentenceTransformer
        self.mock_embedding_model_instance = MockSentenceTransformer.return_value
        self.mock_embedding_model_instance.encode.return_value = np.array([[0.1, 0.2, 0.3]])

        # Mock os.path.exists for load_index called during init
        # It needs to return False initially for some tests, or True for others.
        # We'll handle specific os.path.exists mocks within each test or a more specific setup.
        with patch('llm_integrations.rag_processor.os.path.exists', return_value=False):
            self.rag_processor = RAGProcessor(
                document_dir="test_docs",
                index_path="test_data/test_index.idx",
                embedding_model_name="test-model"
            )
        
        # Reset faiss mock calls for each test
        faiss.reset_mock()


    def test_initialization(self):
        """Test RAGProcessor initialization."""
        self.assertEqual(self.rag_processor.document_dir, "test_docs")
        self.assertEqual(self.rag_processor.index_path, "test_data/test_index.idx")
        self.assertEqual(self.rag_processor.chunks_path, "test_data/test_index_chunks.json")
        self.assertEqual(self.rag_processor.embedding_model_name, "test-model")
        
        # Check that SentenceTransformer was called
        self.mock_embedding_model_instance._mock_parent.assert_called_once_with("test-model")
        
        # Test that load_index was called (implicitly, by checking if index is None or files were checked)
        # This is tricky because load_index is complex. We'll test load_index separately.
        # For now, we know it's called. If it found nothing (default mock for os.path.exists), index is None.
        self.assertIsNone(self.rag_processor.index)
        self.assertEqual(self.rag_processor.document_chunks, [])
        print("TestRAGProcessor: test_initialization PASSED")

    @patch('llm_integrations.rag_processor.os.listdir')
    @patch('llm_integrations.rag_processor.os.path.join', side_effect=lambda *args: "/".join(args)) # Simple join
    @patch('llm_integrations.rag_processor.open', new_callable=mock_open)
    @patch('llm_integrations.rag_processor.os.path.exists') # For document_dir check
    def test_load_and_chunk_documents_success(self, mock_dir_exists, mock_file_open, mock_os_join, mock_os_listdir):
        """Test successful loading and chunking of documents."""
        mock_dir_exists.return_value = True # document_dir exists
        mock_os_listdir.return_value = ["doc1.txt", "doc2.md", "other.file"]
        
        # Configure mock_open to return different content for different files
        mock_file_open.side_effect = [
            mock_open(read_data="Content of doc1.\n\nParagraph 2 of doc1.").return_value, # doc1.txt
            mock_open(read_data="# Title\n\nMD Content para1.\n\nMD Content para2.").return_value, # doc2.md
        ]

        chunks = self.rag_processor._load_and_chunk_documents()
        
        expected_chunks = [
            "Content of doc1.", "Paragraph 2 of doc1.",
            "# Title", "MD Content para1.", "MD Content para2."
        ]
        self.assertEqual(chunks, expected_chunks)
        
        # Check calls to open
        self.assertEqual(mock_file_open.call_count, 2)
        mock_file_open.assert_any_call("test_docs/doc1.txt", 'r', encoding='utf-8')
        mock_file_open.assert_any_call("test_docs/doc2.md", 'r', encoding='utf-8')
        print("TestRAGProcessor: test_load_and_chunk_documents_success PASSED")

    @patch('llm_integrations.rag_processor.os.path.exists')
    def test_load_and_chunk_documents_no_dir(self, mock_dir_exists):
        """Test _load_and_chunk_documents when document directory does not exist."""
        mock_dir_exists.return_value = False
        chunks = self.rag_processor._load_and_chunk_documents()
        self.assertEqual(chunks, [])
        print("TestRAGProcessor: test_load_and_chunk_documents_no_dir PASSED")

    @patch('llm_integrations.rag_processor.os.listdir', return_value=[])
    @patch('llm_integrations.rag_processor.os.path.exists', return_value=True)
    def test_load_and_chunk_documents_empty_dir(self, mock_dir_exists, mock_os_listdir):
        """Test _load_and_chunk_documents with an empty directory."""
        chunks = self.rag_processor._load_and_chunk_documents()
        self.assertEqual(chunks, [])
        print("TestRAGProcessor: test_load_and_chunk_documents_empty_dir PASSED")

    @patch('llm_integrations.rag_processor.RAGProcessor._load_and_chunk_documents')
    @patch('llm_integrations.rag_processor.json.dump')
    @patch('llm_integrations.rag_processor.os.makedirs') # Mock makedirs
    def test_build_index_success(self, mock_makedirs, mock_json_dump, mock_load_chunks):
        """Test successful building of the FAISS index."""
        mock_load_chunks.return_value = ["chunk1", "chunk2", "chunk3"]
        
        # Mock embeddings from the model
        mock_embeddings = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]], dtype=np.float32)
        self.mock_embedding_model_instance.encode.return_value = mock_embeddings

        # Mock FAISS
        mock_faiss_index_instance = faiss.IndexFlatL2.return_value
        
        self.rag_processor.build_index()

        mock_load_chunks.assert_called_once()
        self.mock_embedding_model_instance.encode.assert_called_once_with(
            ["chunk1", "chunk2", "chunk3"], convert_to_tensor=False, show_progress_bar=True
        )
        faiss.IndexFlatL2.assert_called_once_with(2) # Dimension from embeddings
        mock_faiss_index_instance.add.assert_called_once()
        # Check if np.array_equal can be used for the faiss add call's argument
        np.testing.assert_array_equal(mock_faiss_index_instance.add.call_args[0][0], mock_embeddings)

        faiss.write_index.assert_called_once_with(mock_faiss_index_instance, "test_data/test_index.idx")
        
        # Check json.dump call (it's called with a file object)
        self.assertEqual(mock_json_dump.call_args[0][0], ["chunk1", "chunk2", "chunk3"])
        
        self.assertIsNotNone(self.rag_processor.index)
        self.assertEqual(self.rag_processor.document_chunks, ["chunk1", "chunk2", "chunk3"])
        mock_makedirs.assert_any_call(os.path.dirname("test_data/test_index.idx"), exist_ok=True)
        mock_makedirs.assert_any_call(os.path.dirname("test_data/test_index_chunks.json"), exist_ok=True)

        print("TestRAGProcessor: test_build_index_success PASSED")

    @patch('llm_integrations.rag_processor.RAGProcessor._load_and_chunk_documents', return_value=[])
    @patch('llm_integrations.rag_processor.os.remove')
    def test_build_index_no_documents(self, mock_os_remove, mock_load_chunks):
        """Test build_index when no documents are found."""
        with patch('llm_integrations.rag_processor.os.path.exists', return_value=True): # Simulate old files exist
             self.rag_processor.build_index()

        mock_load_chunks.assert_called_once()
        self.mock_embedding_model_instance.encode.assert_not_called()
        faiss.IndexFlatL2.assert_not_called()
        self.assertIsNone(self.rag_processor.index)
        self.assertEqual(self.rag_processor.document_chunks, [])
        
        # Check if os.remove was called for old index files
        expected_remove_calls = [
            call("test_data/test_index.idx"),
            call("test_data/test_index_chunks.json")
        ]
        mock_os_remove.assert_has_calls(expected_remove_calls, any_order=True)
        print("TestRAGProcessor: test_build_index_no_documents PASSED")


    @patch('llm_integrations.rag_processor.os.path.exists')
    @patch('llm_integrations.rag_processor.json.load')
    def test_load_index_success(self, mock_json_load, mock_path_exists):
        """Test successful loading of an existing index."""
        mock_path_exists.side_effect = lambda path: True # All paths exist
        
        mock_faiss_index_instance = MagicMock()
        mock_faiss_index_instance.ntotal = 2
        faiss.read_index.return_value = mock_faiss_index_instance
        
        mock_chunks = ["chunkA", "chunkB"]
        mock_json_load.return_value = mock_chunks
        
        self.rag_processor.load_index()
        
        faiss.read_index.assert_called_once_with("test_data/test_index.idx")
        mock_json_load.assert_called_once() # With a file object
        self.assertIs(self.rag_processor.index, mock_faiss_index_instance)
        self.assertEqual(self.rag_processor.document_chunks, mock_chunks)
        print("TestRAGProcessor: test_load_index_success PASSED")

    @patch('llm_integrations.rag_processor.os.path.exists')
    @patch('llm_integrations.rag_processor.json.load')
    def test_load_index_mismatch(self, mock_json_load, mock_path_exists):
        """Test loading index with a mismatch between index size and chunk count."""
        mock_path_exists.return_value = True
        
        mock_faiss_index_instance = MagicMock()
        mock_faiss_index_instance.ntotal = 5 # Index says 5 items
        faiss.read_index.return_value = mock_faiss_index_instance
        
        mock_chunks = ["chunkA", "chunkB"] # But only 2 chunks loaded
        mock_json_load.return_value = mock_chunks
        
        # Capture print output to check for warning
        with patch('builtins.print') as mock_print:
            self.rag_processor.load_index()
        
        self.assertIs(self.rag_processor.index, mock_faiss_index_instance)
        self.assertEqual(self.rag_processor.document_chunks, mock_chunks)
        mock_print.assert_any_call("Warning: Mismatch between FAISS index size (5) and number of loaded chunks (2). The index may be stale. Consider rebuilding.")
        print("TestRAGProcessor: test_load_index_mismatch PASSED")

    @patch('llm_integrations.rag_processor.os.path.exists', return_value=False)
    def test_load_index_files_not_found(self, mock_path_exists):
        """Test load_index when index or chunks files are missing."""
        self.rag_processor.load_index()
        faiss.read_index.assert_not_called()
        self.assertIsNone(self.rag_processor.index)
        self.assertEqual(self.rag_processor.document_chunks, [])
        print("TestRAGProcessor: test_load_index_files_not_found PASSED")


    def test_retrieve_context_success(self):
        """Test successful context retrieval."""
        # Setup mocked index and chunks
        self.rag_processor.index = MagicMock()
        self.rag_processor.index.ntotal = 2
        self.rag_processor.document_chunks = ["Retrieved chunk 1", "Retrieved chunk 2"]
        
        # Mock query embedding
        query_embedding = np.array([[0.4, 0.5, 0.6]], dtype=np.float32)
        self.mock_embedding_model_instance.encode.return_value = query_embedding
        
        # Mock FAISS search
        # Search returns (distances, indices)
        mock_distances = np.array([[0.1, 0.2]])
        mock_indices = np.array([[1, 0]]) # Retrieve chunk 2 then chunk 1
        self.rag_processor.index.search.return_value = (mock_distances, mock_indices)
        
        context = self.rag_processor.retrieve_context("Test query", top_k=2)
        
        self.mock_embedding_model_instance.encode.assert_called_once_with(["Test query"], convert_to_tensor=False)
        self.rag_processor.index.search.assert_called_once()
        
        # Check the first argument of search (query_embedding) with np.testing
        np.testing.assert_array_equal(self.rag_processor.index.search.call_args[0][0], query_embedding)
        self.assertEqual(self.rag_processor.index.search.call_args[0][1], 2) # top_k

        expected_context = "Retrieved chunk 2\n\n---\n\nRetrieved chunk 1"
        self.assertEqual(context, expected_context)
        print("TestRAGProcessor: test_retrieve_context_success PASSED")

    def test_retrieve_context_index_not_ready(self):
        """Test retrieve_context when index is not ready."""
        self.rag_processor.index = None
        context = self.rag_processor.retrieve_context("Test query")
        self.assertEqual(context, "RAG Index not available or empty. Please build or load it first.")
        
        self.rag_processor.index = MagicMock()
        self.rag_processor.index.ntotal = 0 # Empty index
        self.rag_processor.document_chunks = []
        context = self.rag_processor.retrieve_context("Test query")
        self.assertEqual(context, "RAG Index is empty. No context to retrieve.")
        print("TestRAGProcessor: test_retrieve_context_index_not_ready PASSED")

    def test_retrieve_context_top_k_adjustment(self):
        """Test top_k adjustment if it's larger than index size."""
        self.rag_processor.index = MagicMock()
        self.rag_processor.index.ntotal = 1 # Only one item in index
        self.rag_processor.document_chunks = ["Only one chunk"]
        
        self.mock_embedding_model_instance.encode.return_value = np.array([[0.1, 0.1]], dtype=np.float32)
        self.rag_processor.index.search.return_value = (np.array([[0.05]]), np.array([[0]]))
        
        with patch('builtins.print') as mock_print:
            context = self.rag_processor.retrieve_context("Query", top_k=5)
        
        self.rag_processor.index.search.assert_called_once()
        # Check that search was called with adjusted top_k=1
        self.assertEqual(self.rag_processor.index.search.call_args[0][1], 1) 
        mock_print.assert_any_call("Warning: Requested top_k (5) is greater than the number of items in the index (1). Using top_k=1.")
        self.assertEqual(context, "Only one chunk")
        print("TestRAGProcessor: test_retrieve_context_top_k_adjustment PASSED")

    def test_retrieve_context_embedding_model_not_loaded(self):
        """Test context retrieval when embedding model failed to load."""
        self.rag_processor.embedding_model = None # Simulate embedding model load failure
        context = self.rag_processor.retrieve_context("Test query")
        self.assertEqual(context, "Error: Embedding model is not available. Cannot retrieve context.")
        print("TestRAGProcessor: test_retrieve_context_embedding_model_not_loaded PASSED")


if __name__ == '__main__':
    unittest.main(verbosity=2)
