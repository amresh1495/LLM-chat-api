import unittest
from unittest.mock import patch, MagicMock
import torch # Ensure torch is imported as it's used in the clients

# Attempt to import clients; handle potential errors if files don't exist or have issues
try:
    from llm_integrations.gemma_client import GemmaClient
    from llm_integrations.llama_client import LlamaClient
except ImportError as e:
    # This allows tests to be discovered and run even if client files are problematic,
    # though specific tests for those clients might fail.
    print(f"Could not import LLM clients for testing: {e}. Some tests may fail.")
    GemmaClient = None 
    LlamaClient = None


class TestGemmaClient(unittest.TestCase):
    def setUp(self):
        # Common setup for Gemma client tests, if any
        pass

    @patch('llm_integrations.gemma_client.pipeline')
    def test_gemma_client_init_success(self, mock_pipeline):
        """Test successful initialization of GemmaClient."""
        if GemmaClient is None: self.skipTest("GemmaClient not imported.")
        
        mock_pipeline_instance = MagicMock()
        mock_pipeline.return_value = mock_pipeline_instance
        
        client = GemmaClient(model_id="test-gemma-model")
        self.assertIsNotNone(client)
        self.assertEqual(client.model_id, "test-gemma-model")
        mock_pipeline.assert_called_once_with(
            "image-text-to-text",
            model="test-gemma-model",
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        print("TestGemmaClient: test_gemma_client_init_success PASSED")

    @patch('llm_integrations.gemma_client.pipeline')
    def test_gemma_client_init_failure(self, mock_pipeline):
        """Test GemmaClient initialization failure."""
        if GemmaClient is None: self.skipTest("GemmaClient not imported.")

        mock_pipeline.side_effect = RuntimeError("Model loading failed")
        
        with self.assertRaises(RuntimeError) as context:
            GemmaClient(model_id="test-gemma-fail-model")
        self.assertTrue("Failed to load Gemma model" in str(context.exception))
        self.assertTrue("Model loading failed" in str(context.exception))
        mock_pipeline.assert_called_once_with(
            "image-text-to-text",
            model="test-gemma-fail-model",
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        print("TestGemmaClient: test_gemma_client_init_failure PASSED")

    @patch('llm_integrations.gemma_client.pipeline')
    def test_gemma_generate_response_no_history(self, mock_pipeline):
        """Test GemmaClient generate_response without chat history."""
        if GemmaClient is None: self.skipTest("GemmaClient not imported.")

        mock_pipe_instance = MagicMock()
        # Mock output structure from Gemma "image-text-to-text" pipeline
        # The client expects the final assistant message's content.
        mock_gemma_output = [{
            "generated_text": [
                {"role": "system", "content": [{"type": "text", "text": "You are a helpful AI assistant."}]},
                {"role": "user", "content": [{"type": "text", "text": "Hello Gemma"}]},
                {"role": "assistant", "content": [{"type": "text", "text": "Hi there! This is Gemma."}]}
            ]
        }]
        mock_pipe_instance.return_value = mock_gemma_output
        mock_pipeline.return_value = mock_pipe_instance

        client = GemmaClient(model_id="test-gemma-model")
        response = client.generate_response("Hello Gemma")

        self.assertEqual(response, "Hi there! This is Gemma.")
        
        # Verify the arguments passed to the pipeline
        expected_messages_arg = [
            {"role": "system", "content": [{"type": "text", "text": "You are a helpful AI assistant."}]},
            {"role": "user", "content": [{"type": "text", "text": "Hello Gemma"}]}
        ]
        mock_pipe_instance.assert_called_once()
        args, kwargs = mock_pipe_instance.call_args
        self.assertEqual(kwargs.get('text'), expected_messages_arg)
        self.assertEqual(kwargs.get('max_new_tokens'), 500)
        print("TestGemmaClient: test_gemma_generate_response_no_history PASSED")


    @patch('llm_integrations.gemma_client.pipeline')
    def test_gemma_generate_response_with_history(self, mock_pipeline):
        """Test GemmaClient generate_response with chat history."""
        if GemmaClient is None: self.skipTest("GemmaClient not imported.")

        mock_pipe_instance = MagicMock()
        mock_gemma_output = [{
            "generated_text": [
                {"role": "system", "content": [{"type": "text", "text": "You are a helpful AI assistant."}]},
                {"role": "user", "content": [{"type": "text", "text": "Past question?"}]},
                {"role": "assistant", "content": [{"type": "text", "text": "Past answer."}]},
                {"role": "user", "content": [{"type": "text", "text": "New question?"}]},
                {"role": "assistant", "content": [{"type": "text", "text": "New answer!"}]}
            ]
        }]
        mock_pipe_instance.return_value = mock_gemma_output
        mock_pipeline.return_value = mock_pipe_instance

        client = GemmaClient(model_id="test-gemma-model")
        chat_history = [
            # GemmaClient expects content to be adaptable or already in list-of-dicts format
            # The client code adapts string content from history to `[{"type": "text", "text": "..."}]`
            {"role": "user", "content": "Past question?"}, 
            {"role": "assistant", "content": "Past answer."} 
        ]
        response = client.generate_response("New question?", chat_history=chat_history)
        self.assertEqual(response, "New answer!")

        expected_messages_arg = [
            {"role": "system", "content": [{"type": "text", "text": "You are a helpful AI assistant."}]},
            {"role": "user", "content": [{"type": "text", "text": "Past question?"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "Past answer."}]},
            {"role": "user", "content": [{"type": "text", "text": "New question?"}]}
        ]
        mock_pipe_instance.assert_called_once()
        args, kwargs = mock_pipe_instance.call_args
        self.assertEqual(kwargs.get('text'), expected_messages_arg)
        print("TestGemmaClient: test_gemma_generate_response_with_history PASSED")

    @patch('llm_integrations.gemma_client.pipeline')
    def test_gemma_generate_response_pipeline_error(self, mock_pipeline):
        """Test GemmaClient generate_response when pipeline call fails."""
        if GemmaClient is None: self.skipTest("GemmaClient not imported.")

        mock_pipe_instance = MagicMock()
        mock_pipe_instance.side_effect = RuntimeError("Pipeline execution error")
        mock_pipeline.return_value = mock_pipe_instance
        
        client = GemmaClient(model_id="test-gemma-model")
        with self.assertRaises(RuntimeError) as context:
            client.generate_response("Hello")
        self.assertTrue("Failed to generate response from Gemma" in str(context.exception))
        self.assertTrue("Pipeline execution error" in str(context.exception))
        print("TestGemmaClient: test_gemma_generate_response_pipeline_error PASSED")


class TestLlamaClient(unittest.TestCase):
    def setUp(self):
        # Common setup for Llama client tests, if any
        pass

    @patch('llm_integrations.llama_client.pipeline')
    def test_llama_client_init_success(self, mock_pipeline):
        """Test successful initialization of LlamaClient."""
        if LlamaClient is None: self.skipTest("LlamaClient not imported.")

        mock_pipeline_instance = MagicMock()
        mock_pipeline.return_value = mock_pipeline_instance
        
        client = LlamaClient(model_id="test-llama-model")
        self.assertIsNotNone(client)
        self.assertEqual(client.model_id, "test-llama-model")
        mock_pipeline.assert_called_once_with(
            "text-generation",
            model="test-llama-model",
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        print("TestLlamaClient: test_llama_client_init_success PASSED")

    @patch('llm_integrations.llama_client.pipeline')
    def test_llama_client_init_failure(self, mock_pipeline):
        """Test LlamaClient initialization failure."""
        if LlamaClient is None: self.skipTest("LlamaClient not imported.")

        mock_pipeline.side_effect = RuntimeError("Llama model loading failed")
        
        with self.assertRaises(RuntimeError) as context:
            LlamaClient(model_id="test-llama-fail-model")
        self.assertTrue("Failed to load Llama model" in str(context.exception))
        self.assertTrue("Llama model loading failed" in str(context.exception))
        mock_pipeline.assert_called_once_with(
            "text-generation",
            model="test-llama-fail-model",
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        print("TestLlamaClient: test_llama_client_init_failure PASSED")

    @patch('llm_integrations.llama_client.pipeline')
    def test_llama_generate_response_no_history(self, mock_pipeline):
        """Test LlamaClient generate_response without chat history."""
        if LlamaClient is None: self.skipTest("LlamaClient not imported.")

        mock_pipe_instance = MagicMock()
        # Mock output structure from Llama "text-generation" pipeline
        mock_llama_output = [{
            "generated_text": [ # This is a list of message dicts
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": "Hello Llama"},
                {"role": "assistant", "content": "Hi there! This is Llama."}
            ]
        }]
        mock_pipe_instance.return_value = mock_llama_output
        mock_pipeline.return_value = mock_pipe_instance

        client = LlamaClient(model_id="test-llama-model")
        response = client.generate_response("Hello Llama")
        self.assertEqual(response, "Hi there! This is Llama.")
        
        # Verify arguments passed to the pipeline
        expected_messages_arg = [
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": "Hello Llama"}
        ]
        # The pipeline is called with (messages, max_new_tokens=..., do_sample=..., ...)
        mock_pipe_instance.assert_called_once()
        args, kwargs = mock_pipe_instance.call_args
        self.assertEqual(args[0], expected_messages_arg) # Messages is the first positional arg
        self.assertEqual(kwargs.get('max_new_tokens'), 500)
        print("TestLlamaClient: test_llama_generate_response_no_history PASSED")

    @patch('llm_integrations.llama_client.pipeline')
    def test_llama_generate_response_with_history(self, mock_pipeline):
        """Test LlamaClient generate_response with chat history."""
        if LlamaClient is None: self.skipTest("LlamaClient not imported.")

        mock_pipe_instance = MagicMock()
        mock_llama_output = [{
            "generated_text": [
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": "Old Llama question?"},
                {"role": "assistant", "content": "Old Llama answer."},
                {"role": "user", "content": "New Llama question?"},
                {"role": "assistant", "content": "New Llama answer!"}
            ]
        }]
        mock_pipe_instance.return_value = mock_llama_output
        mock_pipeline.return_value = mock_pipe_instance

        client = LlamaClient(model_id="test-llama-model")
        chat_history = [
            {"role": "user", "content": "Old Llama question?"},
            {"role": "assistant", "content": "Old Llama answer."}
        ]
        response = client.generate_response("New Llama question?", chat_history=chat_history)
        self.assertEqual(response, "New Llama answer!")

        expected_messages_arg = [
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": "Old Llama question?"},
            {"role": "assistant", "content": "Old Llama answer."},
            {"role": "user", "content": "New Llama question?"}
        ]
        mock_pipe_instance.assert_called_once()
        args, kwargs = mock_pipe_instance.call_args
        self.assertEqual(args[0], expected_messages_arg)
        print("TestLlamaClient: test_llama_generate_response_with_history PASSED")

    @patch('llm_integrations.llama_client.pipeline')
    def test_llama_generate_response_pipeline_error(self, mock_pipeline):
        """Test LlamaClient generate_response when pipeline call fails."""
        if LlamaClient is None: self.skipTest("LlamaClient not imported.")

        mock_pipe_instance = MagicMock()
        mock_pipe_instance.side_effect = RuntimeError("Llama pipeline execution error")
        mock_pipeline.return_value = mock_pipe_instance
        
        client = LlamaClient(model_id="test-llama-model")
        with self.assertRaises(RuntimeError) as context:
            client.generate_response("Hello Llama")
        self.assertTrue("Failed to generate response from Llama" in str(context.exception))
        self.assertTrue("Llama pipeline execution error" in str(context.exception))
        print("TestLlamaClient: test_llama_generate_response_pipeline_error PASSED")


if __name__ == '__main__':
    # This allows running the tests directly from the script.
    # You might need to adjust Python's path if llm_integrations is not discoverable.
    # e.g., by running from the parent directory of llm_integrations and tests:
    # python -m unittest tests.test_llm_clients
    unittest.main(verbosity=2)
