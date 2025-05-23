import unittest
from unittest.mock import patch, MagicMock, call
import sys

# Ensure the app directory is in the path for imports if running tests from root
# This might be necessary if 'app' is not automatically discoverable.
# For example, if 'tests' is a top-level directory alongside 'app_directory_name'.
# If your project structure is flat (app.py in root), this might not be needed.
# Assuming app.py is at the root or Python path is configured.
try:
    from app import (
        load_model,
        convert_gradio_history_to_llm_format,
        chat_function,
        clear_chat_and_unload_models,
        initialize_rag_processor, # New import
        build_rag_index_action,   # New import
        GEMMA_MODEL_NAME,
        LLAMA_MODEL_NAME,
        DATA_DIR,                 # New import
        INDEX_PATH                # New import
    )
    # Also need to patch the global client variables within the 'app' module
except ImportError as e:
    print(f"Failed to import from app: {e}. Ensure app.py is in PYTHONPATH.")
    # Define placeholders if import fails, so tests can be discovered (though they will fail)
    load_model = convert_gradio_history_to_llm_format = chat_function = clear_chat_and_unload_models = None
    initialize_rag_processor = build_rag_index_action = None
    GEMMA_MODEL_NAME = "Gemma 3 (12B IT)" 
    LLAMA_MODEL_NAME = "Llama 3.2 (3B Instruct)" 
    DATA_DIR = "data"
    INDEX_PATH = "data/faiss_index.idx"


# To access and modify the global variables gemma_client and llama_client in app.py,
# we need to import the 'app' module itself or patch 'app.gemma_client' and 'app.llama_client'.
# Let's try to import 'app' directly to manipulate its globals.
try:
    import app
except ImportError:
    app = None # Placeholder if app module itself cannot be imported.


class TestAppLogic(unittest.TestCase):

    def setUp(self):
        """
        Reset global states in the 'app' module before each test.
        """
        if app:
            app.gemma_client = None
            app.llama_client = None
            app.rag_processor = None # Reset RAG processor
        # If torch or gc were globally imported in app.py and need reset, handle here.
        # For this test, we primarily mock their usage within functions.

    def tearDown(self):
        """
        Clean up global states in 'app' module after each test if necessary.
        """
        if app:
            app.gemma_client = None
            app.llama_client = None
            app.rag_processor = None # Reset RAG processor
    
    @patch('app.gc.collect')
    @patch('app.torch.cuda.empty_cache')
    @patch('app.LlamaClient')
    @patch('app.GemmaClient')
    def test_load_model_gemma_success(self, MockGemmaClient, MockLlamaClient, mock_empty_cache, mock_gc_collect):
        if not all([load_model, app]): self.skipTest("App module or function not imported.")

        mock_gemma_instance = MockGemmaClient.return_value
        
        status = load_model(GEMMA_MODEL_NAME)
        
        self.assertEqual(status, f"{GEMMA_MODEL_NAME} loaded successfully.")
        self.assertIsNotNone(app.gemma_client)
        self.assertIs(app.gemma_client, mock_gemma_instance)
        MockGemmaClient.assert_called_once()
        MockLlamaClient.assert_not_called()
        mock_empty_cache.assert_not_called() # Not called if other client was None
        mock_gc_collect.assert_not_called()  # Not called if other client was None
        print("TestAppLogic: test_load_model_gemma_success PASSED")

    @patch('app.gc.collect')
    @patch('app.torch.cuda.empty_cache')
    @patch('app.LlamaClient')
    @patch('app.GemmaClient')
    def test_load_model_llama_success(self, MockGemmaClient, MockLlamaClient, mock_empty_cache, mock_gc_collect):
        if not all([load_model, app]): self.skipTest("App module or function not imported.")

        mock_llama_instance = MockLlamaClient.return_value

        status = load_model(LLAMA_MODEL_NAME)

        self.assertEqual(status, f"{LLAMA_MODEL_NAME} loaded successfully.")
        self.assertIsNotNone(app.llama_client)
        self.assertIs(app.llama_client, mock_llama_instance)
        MockLlamaClient.assert_called_once()
        MockGemmaClient.assert_not_called()
        mock_empty_cache.assert_not_called()
        mock_gc_collect.assert_not_called()
        print("TestAppLogic: test_load_model_llama_success PASSED")

    @patch('app.gc.collect')
    @patch('app.torch.cuda.empty_cache')
    @patch('app.LlamaClient')
    @patch('app.GemmaClient')
    def test_load_model_gemma_already_loaded(self, MockGemmaClient, MockLlamaClient, mock_empty_cache, mock_gc_collect):
        if not all([load_model, app]): self.skipTest("App module or function not imported.")
        
        app.gemma_client = MockGemmaClient.return_value # Simulate already loaded

        status = load_model(GEMMA_MODEL_NAME)
        self.assertEqual(status, f"{GEMMA_MODEL_NAME} is already loaded.")
        MockGemmaClient.assert_called_once() # From pre-setting it
        self.assertEqual(MockGemmaClient.call_count, 1) # Not called again
        print("TestAppLogic: test_load_model_gemma_already_loaded PASSED")

    @patch('app.gc.collect')
    @patch('app.torch.cuda.empty_cache')
    @patch('app.LlamaClient')
    @patch('app.GemmaClient')
    def test_load_model_llama_initialization_failure(self, MockGemmaClient, MockLlamaClient, mock_empty_cache, mock_gc_collect):
        if not all([load_model, app]): self.skipTest("App module or function not imported.")

        MockLlamaClient.side_effect = RuntimeError("Llama init failed")
        
        status = load_model(LLAMA_MODEL_NAME)
        
        self.assertTrue("Error loading Llama 3.2 (3B Instruct): Llama init failed" in status)
        self.assertIsNone(app.llama_client)
        mock_empty_cache.assert_called_once() # Called on failure
        mock_gc_collect.assert_called_once()  # Called on failure
        print("TestAppLogic: test_load_model_llama_initialization_failure PASSED")

    @patch('app.gc.collect')
    @patch('app.torch.cuda.empty_cache')
    @patch('app.LlamaClient')
    @patch('app.GemmaClient')
    def test_load_model_switch_from_gemma_to_llama(self, MockGemmaClient, MockLlamaClient, mock_empty_cache, mock_gc_collect):
        if not all([load_model, app]): self.skipTest("App module or function not imported.")

        # Pre-load Gemma
        mock_gemma_instance = MockGemmaClient.return_value
        app.gemma_client = mock_gemma_instance

        # Load Llama
        mock_llama_instance = MockLlamaClient.return_value
        status = load_model(LLAMA_MODEL_NAME)

        self.assertEqual(status, f"{LLAMA_MODEL_NAME} loaded successfully.")
        self.assertIsNone(app.gemma_client, "Gemma client should be unloaded")
        self.assertIsNotNone(app.llama_client)
        self.assertIs(app.llama_client, mock_llama_instance)
        
        MockLlamaClient.assert_called_once()
        mock_empty_cache.assert_called_once() # Called when unloading Gemma
        mock_gc_collect.assert_called_once()  # Called when unloading Gemma
        print("TestAppLogic: test_load_model_switch_from_gemma_to_llama PASSED")


    def test_convert_gradio_history_to_llm_format(self):
        if convert_gradio_history_to_llm_format is None: self.skipTest("Function not imported.")

        # Test with empty history
        gradio_history_empty = []
        expected_llm_empty = []
        self.assertEqual(convert_gradio_history_to_llm_format(gradio_history_empty), expected_llm_empty)

        # Test with some interactions
        gradio_history_full = [
            ["Hi", "Hello there!"],
            ["How are you?", "I am fine, thank you."]
        ]
        expected_llm_full = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello there!"},
            {"role": "user", "content": "How are you?"},
            {"role": "assistant", "content": "I am fine, thank you."}
        ]
        self.assertEqual(convert_gradio_history_to_llm_format(gradio_history_full), expected_llm_full)
        
        # Test with only user message in last turn (Gradio might do this before bot responds)
        gradio_history_partial = [
            ["Hi", "Hello there!"],
            ["How are you?", None] # User sent message, bot hasn't responded yet in Gradio's internal list
        ]
        expected_llm_partial = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello there!"},
            {"role": "user", "content": "How are you?"}
            # No assistant message for the last None
        ]
        self.assertEqual(convert_gradio_history_to_llm_format(gradio_history_partial), expected_llm_partial)
        print("TestAppLogic: test_convert_gradio_history_to_llm_format PASSED")

    @patch('app.convert_gradio_history_to_llm_format')
    def test_chat_function_gemma_not_loaded(self, mock_convert_history):
        if not all([chat_function, app]): self.skipTest("App module or function not imported.")
        app.gemma_client = None
        response = chat_function("Hello", [], GEMMA_MODEL_NAME, rag_enabled=False) # Added rag_enabled
        self.assertEqual(response, f"ERROR: {GEMMA_MODEL_NAME} is not loaded. Please load it first.")
        mock_convert_history.assert_called_once() 
        print("TestAppLogic: test_chat_function_gemma_not_loaded PASSED")

    @patch('app.convert_gradio_history_to_llm_format')
    @patch('app.GemmaClient') 
    def test_chat_function_gemma_success_rag_disabled(self, MockGemmaClient, mock_convert_history):
        if not all([chat_function, app]): self.skipTest("App module or function not imported.")

        mock_gemma_instance = MockGemmaClient.return_value
        mock_gemma_instance.generate_response.return_value = "Gemma response"
        app.gemma_client = mock_gemma_instance 
        
        mock_converted_history = [{"role": "user", "content": "Previous message"}]
        mock_convert_history.return_value = mock_converted_history

        response = chat_function("Test message", [["Previous message", "Bot previous"]], GEMMA_MODEL_NAME, rag_enabled=False)
        
        self.assertEqual(response, "Gemma response")
        mock_convert_history.assert_called_once_with([["Previous message", "Bot previous"]])
        mock_gemma_instance.generate_response.assert_called_once_with(
            user_prompt="Test message",
            chat_history=mock_converted_history,
            retrieved_context=None # RAG disabled
        )
        print("TestAppLogic: test_chat_function_gemma_success_rag_disabled PASSED")

    @patch('app.convert_gradio_history_to_llm_format')
    @patch('app.LlamaClient') 
    def test_chat_function_llama_success_rag_disabled(self, MockLlamaClient, mock_convert_history):
        if not all([chat_function, app]): self.skipTest("App module or function not imported.")

        mock_llama_instance = MockLlamaClient.return_value
        mock_llama_instance.generate_response.return_value = "Llama response"
        app.llama_client = mock_llama_instance 

        mock_converted_history = [{"role": "user", "content": "Old message"}]
        mock_convert_history.return_value = mock_converted_history

        response = chat_function("New Llama message", [["Old message", "Old Llama bot"]], LLAMA_MODEL_NAME, rag_enabled=False)

        self.assertEqual(response, "Llama response")
        mock_convert_history.assert_called_once_with([["Old message", "Old Llama bot"]])
        mock_llama_instance.generate_response.assert_called_once_with(
            user_prompt="New Llama message",
            chat_history=mock_converted_history,
            retrieved_context=None # RAG disabled
        )
        print("TestAppLogic: test_chat_function_llama_success_rag_disabled PASSED")

    @patch('app.RAGProcessor') # Mock RAGProcessor at the app level
    @patch('app.convert_gradio_history_to_llm_format')
    @patch('app.GemmaClient')
    def test_chat_function_gemma_rag_enabled_index_ready(self, MockGemmaClient, mock_convert_history, MockRAGProcessor):
        if not all([chat_function, app]): self.skipTest("App module or function not imported.")

        # Setup LLM client
        mock_gemma_instance = MockGemmaClient.return_value
        mock_gemma_instance.generate_response.return_value = "Gemma RAG response"
        app.gemma_client = mock_gemma_instance

        # Setup RAG processor
        mock_rag_instance = MockRAGProcessor.return_value
        mock_rag_instance.index = MagicMock() # Simulate index exists
        mock_rag_instance.document_chunks = ["chunk1"] # Simulate chunks exist
        mock_rag_instance.retrieve_context.return_value = "Retrieved RAG context"
        app.rag_processor = mock_rag_instance # Assign to app's global

        mock_convert_history.return_value = []
        response = chat_function("Query for RAG", [], GEMMA_MODEL_NAME, rag_enabled=True)

        self.assertEqual(response, "Gemma RAG response")
        mock_rag_instance.retrieve_context.assert_called_once_with(query="Query for RAG")
        mock_gemma_instance.generate_response.assert_called_once_with(
            user_prompt="Query for RAG",
            chat_history=[],
            retrieved_context="Retrieved RAG context"
        )
        print("TestAppLogic: test_chat_function_gemma_rag_enabled_index_ready PASSED")

    @patch('app.RAGProcessor')
    @patch('app.convert_gradio_history_to_llm_format')
    @patch('app.GemmaClient')
    def test_chat_function_gemma_rag_enabled_index_not_ready(self, MockGemmaClient, mock_convert_history, MockRAGProcessor):
        if not all([chat_function, app]): self.skipTest("App module or function not imported.")
        
        mock_gemma_instance = MockGemmaClient.return_value
        mock_gemma_instance.generate_response.return_value = "Gemma no RAG response"
        app.gemma_client = mock_gemma_instance

        # RAG processor initialized, but index is None (not ready)
        mock_rag_instance = MockRAGProcessor.return_value
        mock_rag_instance.index = None 
        app.rag_processor = mock_rag_instance

        mock_convert_history.return_value = []
        response = chat_function("Query for RAG", [], GEMMA_MODEL_NAME, rag_enabled=True)

        self.assertEqual(response, "Gemma no RAG response")
        mock_rag_instance.retrieve_context.assert_not_called() # Should not be called if index is None
        mock_gemma_instance.generate_response.assert_called_once_with(
            user_prompt="Query for RAG",
            chat_history=[],
            retrieved_context=None # No context as RAG index was not ready
        )
        print("TestAppLogic: test_chat_function_gemma_rag_enabled_index_not_ready PASSED")
        
    @patch('app.convert_gradio_history_to_llm_format')
    @patch('app.GemmaClient')
    @patch('app.torch.cuda.empty_cache') # Mock for CUDA OOM
    @patch('app.gc.collect')          # Mock for CUDA OOM
    def test_chat_function_gemma_client_exception(self, mock_gc_collect, mock_empty_cache, MockGemmaClient, mock_convert_history):
        if not all([chat_function, app]): self.skipTest("App module or function not imported.")

        mock_gemma_instance = MockGemmaClient.return_value
        mock_gemma_instance.generate_response.side_effect = RuntimeError("Gemma generation failed")
        app.gemma_client = mock_gemma_instance
        
        mock_convert_history.return_value = []
        
        response = chat_function("Hello", [], GEMMA_MODEL_NAME)
        
        self.assertTrue("Error during response generation with Gemma 3 (12B IT): Gemma generation failed" in response)
        mock_empty_cache.assert_not_called() # Not called for generic error
        mock_gc_collect.assert_not_called()
        print("TestAppLogic: test_chat_function_gemma_client_exception PASSED")

    @patch('app.convert_gradio_history_to_llm_format')
    @patch('app.LlamaClient')
    @patch('app.torch.cuda.empty_cache')
    @patch('app.gc.collect')
    def test_chat_function_llama_cuda_oom_exception(self, mock_gc_collect, mock_empty_cache, MockLlamaClient, mock_convert_history):
        if not all([chat_function, app]): self.skipTest("App module or function not imported.")

        mock_llama_instance = MockLlamaClient.return_value
        mock_llama_instance.generate_response.side_effect = RuntimeError("CUDA out of memory")
        app.llama_client = mock_llama_instance

        mock_convert_history.return_value = []
        response = chat_function("Large prompt", [], LLAMA_MODEL_NAME)

        self.assertTrue("CUDA out of memory" in response)
        self.assertTrue("Try clearing chat and unloading models" in response) # Check for advice
        mock_empty_cache.assert_called_once() # Called for CUDA OOM
        mock_gc_collect.assert_called_once()  # Called for CUDA OOM
        print("TestAppLogic: test_chat_function_llama_cuda_oom_exception PASSED")


    @patch('app.gc.collect')
    @patch('app.torch.cuda.empty_cache')
    @patch('app.GemmaClient') # Mock to simulate it being loaded
    @patch('app.LlamaClient') # Mock to simulate it being loaded
    def test_clear_chat_and_unload_models(self, MockLlamaClient, MockGemmaClient, mock_empty_cache, mock_gc_collect):
        if not all([clear_chat_and_unload_models, app]): self.skipTest("App module or function not imported.")

        # Simulate models being loaded
        app.gemma_client = MockGemmaClient.return_value
        app.llama_client = MockLlamaClient.return_value
        
        # Patch torch.cuda.is_available to return True to test cache clearing path
        with patch('app.torch.cuda.is_available', return_value=True):
            chat_history_cleared, status_msg = clear_chat_and_unload_models()

        self.assertEqual(chat_history_cleared, [])
        self.assertIsNone(app.gemma_client)
        self.assertIsNone(app.llama_client)
        
        mock_empty_cache.assert_called_once()
        mock_gc_collect.assert_called_once()
        
        self.assertIn("Chat cleared.", status_msg)
        self.assertIn(f"{GEMMA_MODEL_NAME} unloaded.", status_msg)
        self.assertIn(f"{LLAMA_MODEL_NAME} unloaded.", status_msg)
        self.assertIn("CUDA cache cleared.", status_msg)
        self.assertIn("Garbage collection run.", status_msg)
        print("TestAppLogic: test_clear_chat_and_unload_models PASSED")

    @patch('app.gc.collect')
    @patch('app.torch.cuda.empty_cache')
    def test_clear_chat_no_models_loaded(self, mock_empty_cache, mock_gc_collect):
        if not all([clear_chat_and_unload_models, app]): self.skipTest("App module or function not imported.")

        # Ensure clients are None initially
        app.gemma_client = None
        app.llama_client = None

        with patch('app.torch.cuda.is_available', return_value=False): # Simulate no CUDA
            chat_history_cleared, status_msg = clear_chat_and_unload_models()
        
        self.assertEqual(chat_history_cleared, [])
        self.assertIsNone(app.gemma_client)
        self.assertIsNone(app.llama_client)
        
        mock_empty_cache.assert_not_called() # Not called if CUDA not available
        mock_gc_collect.assert_called_once()  # gc.collect is always called
        
        self.assertIn("Chat cleared.", status_msg)
        self.assertNotIn(f"{GEMMA_MODEL_NAME} unloaded.", status_msg) # Should not say unloaded
        self.assertNotIn("CUDA cache cleared.", status_msg)
        print("TestAppLogic: test_clear_chat_no_models_loaded PASSED")


if __name__ == '__main__':
    # This allows running the tests directly from the script.
    # Ensure that app.py is in a location where it can be imported,
    # or adjust sys.path as needed if running this script directly.
    # Typically, you'd run tests using `python -m unittest tests.test_app` from root.
    unittest.main(verbosity=2)
