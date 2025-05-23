import gradio as gr
from llm_integrations.gemma_client import GemmaClient
from llm_integrations.llama_client import LlamaClient
import torch
import gc

# Global variables for LLM clients
gemma_client = None
llama_client = None

# Constants for model names
GEMMA_MODEL_NAME = "Gemma 3 (12B IT)"
LLAMA_MODEL_NAME = "Llama 3.2 (3B Instruct)"

def load_model(model_name: str) -> str:
    """
    Loads the selected LLM.
    This function is triggered when the user selects a model and clicks "Load Model".
    """
    global gemma_client, llama_client
    status = ""
    try:
        if model_name == GEMMA_MODEL_NAME:
            if gemma_client is None:
                print(f"Loading {GEMMA_MODEL_NAME}...")
                gemma_client = GemmaClient() # Default model ID is "google/gemma-3-12b-it"
                status = f"{GEMMA_MODEL_NAME} loaded successfully."
                print(status)
                 # Unload Llama if Gemma is loaded to save resources
                if llama_client is not None:
                    del llama_client
                    llama_client = None
                    torch.cuda.empty_cache()
                    gc.collect()
                    print("Unloaded Llama 3.2 to free resources.")
            else:
                status = f"{GEMMA_MODEL_NAME} is already loaded."
                print(status)
        elif model_name == LLAMA_MODEL_NAME:
            if llama_client is None:
                print(f"Loading {LLAMA_MODEL_NAME}...")
                llama_client = LlamaClient() # Default model ID is "meta-llama/Llama-3.2-3B-Instruct"
                status = f"{LLAMA_MODEL_NAME} loaded successfully."
                print(status)
                # Unload Gemma if Llama is loaded to save resources
                if gemma_client is not None:
                    del gemma_client
                    gemma_client = None
                    torch.cuda.empty_cache()
                    gc.collect()
                    print("Unloaded Gemma 3 to free resources.")
            else:
                status = f"{LLAMA_MODEL_NAME} is already loaded."
                print(status)
        else:
            status = "Invalid model selection."
            print(status)
    except Exception as e:
        status = f"Error loading {model_name}: {str(e)}"
        print(status)
        # Ensure client is None if loading failed
        if model_name == GEMMA_MODEL_NAME:
            gemma_client = None
        elif model_name == LLAMA_MODEL_NAME:
            llama_client = None
        torch.cuda.empty_cache()
        gc.collect()
    return status

def convert_gradio_history_to_llm_format(gradio_history: list) -> list:
    """
    Converts Gradio's chat history format to the format expected by LLM clients.
    Gradio history: [["user_msg1", "bot_msg1"], ["user_msg2", "bot_msg2"], ...]
    LLM client history: [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
    """
    llm_history = []
    if gradio_history is None:
        return llm_history
    for user_msg, bot_msg in gradio_history:
        if user_msg:
            llm_history.append({"role": "user", "content": user_msg})
        if bot_msg: # Gemma client expects content to be a list of dicts for multimodal
            # For Llama, content is a string. We need to adjust based on client.
            # For simplicity here, we'll assume string content is fine,
            # but GemmaClient might need `[{"type": "text", "text": bot_msg}]`
            # This will be handled within the client's generate_response or by client-specific conversion.
            # For now, let's pass the string directly. The clients should adapt.
            llm_history.append({"role": "assistant", "content": bot_msg})
    return llm_history

def chat_function(message: str, history: list, model_selection: str) -> str:
    """
    Core chat function called by Gradio.
    Handles model selection, history conversion, and response generation.
    Returns the bot's response string.
    """
    global gemma_client, llama_client

    if not model_selection:
        return "ERROR: Please select a model from the dropdown and click 'Load Selected Model'."

    # Convert Gradio history to the format expected by the LLM clients
    llm_history = convert_gradio_history_to_llm_format(history)
    
    bot_response = ""
    try:
        if model_selection == GEMMA_MODEL_NAME:
            if gemma_client is None:
                return f"ERROR: {GEMMA_MODEL_NAME} is not loaded. Please load it first."
            print(f"Sending to Gemma: '{message}' with history: {llm_history}")
            # GemmaClient's generate_response expects `chat_history` in {"role": ..., "content": ...} format.
            # And its `content` for Gemma 3 should be `[{"type": "text", "text": "..."}]`
            # We need to adapt the llm_history again for Gemma or ensure GemmaClient handles it.
            # For now, assuming GemmaClient can handle string content by wrapping it.
            # The GemmaClient provided earlier has an adaptation for string content in history.
            bot_response = gemma_client.generate_response(user_prompt=message, chat_history=llm_history)
            print(f"Gemma response: {bot_response}")

        elif model_selection == LLAMA_MODEL_NAME:
            if llama_client is None:
                return f"ERROR: {LLAMA_MODEL_NAME} is not loaded. Please load it first."
            print(f"Sending to Llama: '{message}' with history: {llm_history}")
            # LlamaClient's generate_response also expects {"role": ..., "content": ...} format.
            # String content is fine for Llama.
            bot_response = llama_client.generate_response(user_prompt=message, chat_history=llm_history)
            print(f"Llama response: {bot_response}")
        else:
            return "ERROR: Invalid model selected or model not loaded. Please select and load a model."

    except Exception as e:
        error_message = f"Error during response generation with {model_selection}: {str(e)}"
        print(error_message)
        # Check if the error is due to CUDA out of memory
        if "CUDA out of memory" in str(e) or "allocate" in str(e).lower():
            error_message += "\n\nThis might be a CUDA out of memory issue. Try clearing chat and unloading models, or restarting the application."
            # Attempt to clear cache to potentially recover
            torch.cuda.empty_cache()
            gc.collect()
        return f"ERROR: {error_message}"
        
    return bot_response

def clear_chat_and_unload_models():
    """
    Clears the chat, unloads all models, and frees GPU memory.
    """
    global gemma_client, llama_client
    
    status_message = "Chat cleared. "
    
    if gemma_client is not None:
        print("Unloading Gemma model...")
        del gemma_client
        gemma_client = None
        status_message += f"{GEMMA_MODEL_NAME} unloaded. "
        
    if llama_client is not None:
        print("Unloading Llama model...")
        del llama_client
        llama_client = None
        status_message += f"{LLAMA_MODEL_NAME} unloaded. "

    if torch.cuda.is_available():
        print("Clearing CUDA cache...")
        torch.cuda.empty_cache()
        status_message += "CUDA cache cleared. "
        
    print("Running garbage collection...")
    gc.collect()
    status_message += "Garbage collection run."
    
    print(status_message)
    return [], status_message # Clears chatbot and updates status display


# --- Gradio Interface Definition ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("<h1>LLM Chatbot (Gemma 3 & Llama 3.2)</h1>")
    gr.Markdown("Select a model, load it, and then start chatting. Models are loaded on demand and only one model can be loaded at a time to conserve resources.")

    with gr.Row():
        model_selector = gr.Dropdown(
            choices=[GEMMA_MODEL_NAME, LLAMA_MODEL_NAME],
            label="Select Model",
            info="Choose the LLM you want to interact with."
        )
        load_button = gr.Button("Load Selected Model")

    status_display = gr.Textbox(label="Status", interactive=False, placeholder="Model loading status will appear here...")
    
    chatbot = gr.Chatbot(
        label="Conversation",
        height=600,
        bubble_full_width=False,
        avatar_images=(None, "https://img.icons8.com/fluency/48/robot.png") # (user, bot)
    )
    
    msg_textbox = gr.Textbox(
        label="Your Message",
        placeholder="Type your message here and press Enter or click Send...",
        lines=2, # Start with 2 lines, can expand
        scale=7 # Take more width in the row
    )
    
    send_button = gr.Button("Send", scale=1) # Smaller button

    clear_button = gr.Button("Clear Chat & Unload Models")

    # --- Event Handling ---
    load_button.click(
        fn=load_model,
        inputs=model_selector,
        outputs=status_display
    )

    # Function to handle sending a message (both button click and textbox submit)
    def process_message(message_text, chat_history, current_model_selection):
        if not message_text.strip(): # Do not send empty messages
            return chat_history # Return original history if message is empty
        # Gradio automatically appends user message to history if fn returns only bot response
        # The chat_function is expected to return only the bot's response.
        # Gradio updates the chatbot UI by taking the user input from msg_textbox,
        # appending it to the history, then appending the bot's response.
        return chat_function(message_text, chat_history, current_model_selection)

    # Connect send button
    send_button.click(
        fn=chat_function, # chat_function will return only the bot's response
        inputs=[msg_textbox, chatbot, model_selector],
        outputs=chatbot # Gradio updates chatbot with user msg + bot response
    ).then(lambda: "", outputs=[msg_textbox]) # Clear textbox after send

    # Connect textbox submit (pressing Enter)
    msg_textbox.submit(
        fn=chat_function,
        inputs=[msg_textbox, chatbot, model_selector],
        outputs=chatbot
    ).then(lambda: "", outputs=[msg_textbox]) # Clear textbox after send

    clear_button.click(
        fn=clear_chat_and_unload_models,
        inputs=None,
        outputs=[chatbot, status_display] # Clears chatbot and updates status
    )

if __name__ == "__main__":
    print("Launching Gradio app...")
    # Share=True for public link (optional, remove if not needed for local testing)
    # Set debug=True for more detailed error messages during development.
    demo.launch(debug=True)
    # demo.launch() # For production or when debugging is not needed.
