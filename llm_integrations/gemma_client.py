import torch
from transformers import pipeline

class GemmaClient:
    """
    A client for interacting with Gemma language models using the Hugging Face transformers library.
    """
    def __init__(self, model_id: str = "google/gemma-3-12b-it"):
        """
        Initializes the GemmaClient with a specified model ID.

        Args:
            model_id (str): The Hugging Face model ID for the Gemma model to be used.
                            Defaults to "google/gemma-3-12b-it".

        Raises:
            RuntimeError: If the model pipeline cannot be initialized.
        """
        self.model_id = model_id
        try:
            # Initialize the transformers pipeline.
            # Gemma 3 is multimodal, so "image-text-to-text" is a suitable pipeline,
            # though we will primarily use its text capabilities here.
            # Using torch.bfloat16 for memory efficiency and device_map="auto" for automatic device placement.
            self.pipe = pipeline(
                "image-text-to-text", # Task for multimodal models like Gemma 3
                model=self.model_id,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
            print(f"Gemma model '{self.model_id}' loaded successfully.")
        except Exception as e:
            print(f"Error initializing Gemma model '{self.model_id}': {e}")
            raise RuntimeError(f"Failed to load Gemma model '{self.model_id}': {e}")

    def generate_response(self, user_prompt: str, chat_history: list = None) -> str:
        """
        Generates a response from the Gemma model based on the user prompt and chat history.

        Args:
            user_prompt (str): The user's input prompt.
            chat_history (list, optional): A list of dictionaries representing the conversation history.
                                           Each dictionary should have "role" ("user" or "assistant")
                                           and "content" (the message text, which should be a list
                                           of content blocks, e.g., [{"type": "text", "text": "..."}]).
                                           Defaults to None.

        Returns:
            str: The generated text response from the model.

        Raises:
            RuntimeError: If an error occurs during response generation.
        """
        if chat_history is None:
            chat_history = []

        # Construct the messages list in the format expected by Gemma instruct models.
        messages = [{"role": "system", "content": [{"type": "text", "text": "You are a helpful AI assistant."}]}]

        # Add chat history
        for message_entry in chat_history:
            # Ensure content is in the expected list-of-dicts format
            if isinstance(message_entry["content"], str):
                 messages.append({"role": message_entry["role"], "content": [{"type": "text", "text": message_entry["content"]}]})
            elif isinstance(message_entry["content"], list):
                 messages.append({"role": message_entry["role"], "content": message_entry["content"]})
            else:
                raise ValueError(f"Chat history message content is not a string or list: {message_entry['content']}")


        # Add current user prompt
        messages.append({"role": "user", "content": [{"type": "text", "text": user_prompt}]})

        try:
            # Generate response using the pipeline
            # The pipeline expects 'text' to be the list of messages for this task
            # For "image-text-to-text", the input is typically a dictionary like `{"text": messages, "images": [pil_image]}`
            # Since we are focusing on text, we pass only the text component.
            # The pipeline should handle this gracefully.
            # The model's default tokenizer will format the messages appropriately.
            
            # The input to an "image-text-to-text" pipeline for text-only is typically just the text prompt
            # or a list of messages if the underlying model supports chat.
            # However, the task definition specifies `messages` to be formatted for Gemma instruct.
            # The Hugging Face pipeline for multimodal models usually takes `prompt` or `text` as a string,
            # or a more complex dict for multimodal inputs.
            # Let's try passing the formatted messages directly to the pipeline.
            # If the model's processor/tokenizer expects a specific format, it should handle it.
            # The task description asks for `self.pipe(text=messages, ...)`.
            # The `text` argument in the pipeline is often a single string for image captioning
            # or VQA. For conversational text with a multimodal model, the exact input format
            # can be model-specific. We'll follow the task's instruction to pass `messages` to `text`.

            # The pipeline for "image-text-to-text" might not directly accept a chat format via `text=messages`.
            # It's more common for "text-generation" or "conversational" pipelines.
            # Let's check how Gemma's pipeline expects chat.
            # If `self.pipe.tokenizer.chat_template` is available, that's preferred.
            # Otherwise, we might need to format it as a single string.
            
            # Given the task specified "image-text-to-text" and the messages format,
            # there might be a slight mismatch if the pipeline expects a flat string for `text`.
            # However, some models process chat messages internally.

            # Let's assume the pipeline can process the list of dicts for now, as per task structure.
            # If this fails, one would typically use `tokenizer.apply_chat_template`.
            
            # The task specifies: `self.pipe(text=messages, max_new_tokens=500)`
            # The `text` parameter for image-text-to-text is typically a string prompt.
            # This might be an area to refine if errors occur, potentially by using
            # `self.pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)`
            # to create the input string if the pipeline doesn't directly consume the chat list.
            # For now, adhering to the explicit instruction.

            outputs = self.pipe(
                text=messages, # As specified by the subtask description
                max_new_tokens=500, # Adjust as needed
                do_sample=True,
                temperature=0.7,
                top_k=50,
                top_p=0.95
            )

            # Extract the generated text.
            # The output structure for "image-text-to-text" can be a list of generated sequences.
            # The subtask specifies: output[0]["generated_text"][-1]["content"]
            # This structure `[-1]["content"]` is typical for chat outputs from `text-generation` pipelines
            # when the input is a list of messages and the pipeline returns the full conversation.
            # For `image-text-to-text`, the output might be simpler, e.g., `output[0]["generated_text"]`.

            # Let's check the output structure carefully.
            # If `text` was a list of messages, and the pipeline processed it as a conversation:
            if isinstance(outputs, list) and outputs and isinstance(outputs[0].get("generated_text"), list):
                # This implies the pipeline returned a conversation history
                generated_part = outputs[0]["generated_text"][-1] # This should be the assistant's last message dict
                if isinstance(generated_part, dict) and "content" in generated_part:
                    # Ensure content is text, as Gemma is primarily text-based in this interaction
                    if isinstance(generated_part["content"], list) and generated_part["content"] and "text" in generated_part["content"][0]:
                         response_text = generated_part["content"][0]["text"]
                    elif isinstance(generated_part["content"], str): # Simpler string content
                         response_text = generated_part["content"]
                    else: # Fallback or error for unexpected content structure
                        print(f"Unexpected content structure in generated_part: {generated_part['content']}")
                        raise RuntimeError("Unexpected content structure in model response.")
                else:
                    # If the last part is not a dict with 'content', it might be a flat string (less likely for chat)
                    # or an error/unexpected format.
                    print(f"Unexpected format for generated_part: {generated_part}")
                    raise RuntimeError("Model response format is not as expected (missing 'content' or not a dict).")
            elif isinstance(outputs, list) and outputs and isinstance(outputs[0].get("generated_text"), str):
                # This implies the pipeline returned a flat string response
                response_text = outputs[0]["generated_text"]
            else:
                print(f"Unexpected output structure from pipeline: {outputs}")
                raise RuntimeError("Model response format is not as expected.")
            
            return response_text

        except Exception as e:
            print(f"Error during Gemma response generation: {e}")
            # print(f"Pipeline output structure (if available): {outputs if 'outputs' in locals() else 'N/A'}")
            raise RuntimeError(f"Failed to generate response from Gemma: {e}")

if __name__ == '__main__':
    # Example usage (requires a Hugging Face token with access to Gemma models)
    # Ensure you are logged in via `huggingface-cli login` if using gated models.
    try:
        print("Attempting to initialize GemmaClient...")
        # Using a smaller, more accessible Gemma model for testing if resources are a concern.
        # client = GemmaClient(model_id="google/gemma-2b-it") # Smaller model for testing
        client = GemmaClient() # Defaults to "google/gemma-3-12b-it"

        print("\nGemmaClient initialized. Sending a test prompt...")
        prompt = "Hello, Gemma! Can you tell me a fun fact about the Roman Empire?"
        
        # Example chat history, ensuring content is in the list-of-dicts format for multimodal models
        chat_history_example = [
            {"role": "user", "content": [{"type": "text", "text": "What is the largest desert in the world?"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "The largest desert in the world is the Antarctic Polar Desert."}]}
        ]
        
        print(f"\nUser Prompt: {prompt}")
        print(f"Chat History: {chat_history_example}")

        response = client.generate_response(prompt, chat_history=chat_history_example)
        print(f"\nGemma's Response: {response}")

        print("\nSending another prompt (no history)...")
        prompt_2 = "What are some interesting applications of AI in healthcare?"
        print(f"\nUser Prompt: {prompt_2}")
        response_2 = client.generate_response(prompt_2)
        print(f"\nGemma's Response: {response_2}")

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
            print("Cleaned up Gemma pipeline and CUDA cache (if applicable).")
