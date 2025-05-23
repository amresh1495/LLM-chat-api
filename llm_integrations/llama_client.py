import torch
from transformers import pipeline

class LlamaClient:
    """
    A client for interacting with Llama language models using the Hugging Face transformers library.
    """
    def __init__(self, model_id: str = "meta-llama/Llama-3.2-3B-Instruct"):
        """
        Initializes the LlamaClient with a specified model ID.

        Args:
            model_id (str): The Hugging Face model ID for the Llama model to be used.
                            Defaults to "meta-llama/Llama-3.2-3B-Instruct".

        Raises:
            RuntimeError: If the model pipeline cannot be initialized.
        """
        self.model_id = model_id
        try:
            # Initialize the transformers pipeline for text generation.
            # Using torch.bfloat16 for memory efficiency and device_map="auto" for automatic device placement.
            self.pipe = pipeline(
                "text-generation",
                model=self.model_id,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
            print(f"Llama model '{self.model_id}' loaded successfully.")
        except Exception as e:
            print(f"Error initializing Llama model '{self.model_id}': {e}")
            # It's good practice to provide the original exception for better debugging.
            raise RuntimeError(f"Failed to load Llama model '{self.model_id}': {e}")

    def generate_response(self, user_prompt: str, chat_history: list = None, retrieved_context: str = None) -> str:
        """
        Generates a response from the Llama model based on the user prompt, chat history, and optional retrieved context.

        Args:
            user_prompt (str): The user's input prompt.
            chat_history (list, optional): A list of dictionaries representing the conversation history.
                                           Each dictionary should have "role" ("user" or "assistant")
                                           and "content" (the message text). Defaults to None.
            retrieved_context (str, optional): Context retrieved from RAG to prepend to the user prompt.
                                               Defaults to None.

        Returns:
            str: The generated text response from the model.

        Raises:
            RuntimeError: If an error occurs during response generation.
        """
        if chat_history is None:
            chat_history = []

        # Construct the messages list in the format expected by Llama instruct models.
        # Standard format includes a system message, followed by alternating user/assistant messages.
        messages = [{"role": "system", "content": "You are a helpful AI assistant."}]

        for message_entry in chat_history:
            messages.append({"role": message_entry["role"], "content": message_entry["content"]})

        # Prepare the final user prompt with context if available
        final_user_prompt = user_prompt
        if retrieved_context and retrieved_context.strip():
            final_user_prompt = f"Based on the following context:\n{retrieved_context}\n\nUser query: {user_prompt}"
            print(f"LlamaClient: Using RAG context. Final prompt starts with: '{final_user_prompt[:200]}...'")

        messages.append({"role": "user", "content": final_user_prompt})

        try:
            # Generate response using the pipeline.
            # The pipeline for "text-generation" with instruct-tuned models like Llama 3
            # typically expects a list of message dictionaries.
            # The tokenizer associated with the model will apply the correct chat template.
            outputs = self.pipe(
                messages,
                max_new_tokens=500,  # Max tokens for the generated response
                do_sample=True,      # Enable sampling for more varied responses
                temperature=0.7,     # Controls randomness (0.0 for deterministic, >1.0 for more random)
                top_p=0.9,           # Nucleus sampling: considers the smallest set of tokens whose cumulative probability exceeds top_p
                top_k=50,            # Considers the top K most likely tokens at each step
            )

            # Extract the generated text.
            # The output from the text-generation pipeline when given a list of messages
            # is usually a list containing a dictionary.
            # The generated text is typically the content of the last message in the 'generated_text' list.
            # Example: outputs[0]['generated_text'][-1]['content']
            if outputs and isinstance(outputs, list) and isinstance(outputs[0], dict) and "generated_text" in outputs[0]:
                generated_messages = outputs[0]["generated_text"]
                if isinstance(generated_messages, list) and generated_messages:
                    # The last message in the list is the assistant's response.
                    last_message = generated_messages[-1]
                    if isinstance(last_message, dict) and "content" in last_message:
                        response_text = last_message["content"]
                    else:
                        # Fallback if the structure is not as expected (e.g. if generated_messages is not a list of dicts)
                        # This might happen if the pipeline returns a flat string for simpler inputs,
                        # but for chat messages, it's usually structured.
                        print(f"Unexpected structure for last_message: {last_message}")
                        response_text = str(last_message) # Convert to string as a fallback
                else:
                    # Fallback if 'generated_text' is not a list or is empty
                    print(f"Unexpected structure or empty 'generated_text': {generated_messages}")
                    # This could also be a direct string if the model/pipeline behaves differently
                    response_text = str(generated_messages)
            else:
                print(f"Unexpected output structure from Llama pipeline: {outputs}")
                raise RuntimeError("Model response format is not as expected.")

            return response_text

        except Exception as e:
            print(f"Error during Llama response generation: {e}")
            # It's helpful to see the structure of `outputs` if an error occurs here for debugging.
            # print(f"Pipeline output structure (if available): {outputs if 'outputs' in locals() else 'N/A'}")
            raise RuntimeError(f"Failed to generate response from Llama: {e}")

if __name__ == '__main__':
    # Example usage:
    # Note: Access to Llama 3 models on Hugging Face might require authentication
    # and agreement to Meta's terms. Ensure you are logged in via `huggingface-cli login`.
    try:
        print("Attempting to initialize LlamaClient...")
        # Using the default model: "meta-llama/Llama-3.2-3B-Instruct"
        # For environments with limited resources, a smaller model from the Llama family
        # or a non-instruct version might be considered if this one is too demanding,
        # though "3B" (3 billion parameters) is relatively small for modern LLMs.
        client = LlamaClient()

        print("\nLlamaClient initialized. Sending a test prompt...")
        prompt = "Hello, Llama! Can you explain the concept of black holes in simple terms?"
        
        chat_history_example = [
            {"role": "user", "content": "What is the capital of Australia?"},
            {"role": "assistant", "content": "The capital of Australia is Canberra."}
        ]
        
        print(f"\nUser Prompt: {prompt}")
        print(f"Chat History: {chat_history_example}")

        response = client.generate_response(prompt, chat_history=chat_history_example)
        print(f"\nLlama's Response: {response}")

        print("\nSending another prompt (no history)...")
        prompt_2 = "What are three key features of Python as a programming language?"
        print(f"\nUser Prompt: {prompt_2}")
        response_2 = client.generate_response(prompt_2)
        print(f"\nLlama's Response: {response_2}")

        print("\nSending prompt with RAG context (no history)...")
        prompt_3 = "How is a fox described here?"
        rag_context_example = "The fox is a clever animal with a bushy tail. It is often found in forests."
        print(f"\nUser Prompt: {prompt_3}")
        print(f"RAG Context: {rag_context_example}")
        response_3 = client.generate_response(prompt_3, retrieved_context=rag_context_example)
        print(f"\nLlama's Response (with RAG): {response_3}")

    except RuntimeError as e:
        print(f"Runtime Error during example usage: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during example usage: {e}")
    finally:
        # Clean up GPU memory if a pipeline was loaded
        if 'client' in locals() and hasattr(client, 'pipe'):
            # Explicitly delete the pipeline object to free resources
            del client.pipe
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("Cleaned up Llama pipeline and CUDA cache (if applicable).")
