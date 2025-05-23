import gradio as gr
from llm_integrations.gemma_client import GemmaClient
from llm_integrations.llama_client import LlamaClient
from llm_integrations.rag_processor import RAGProcessor # Import RAGProcessor
import torch
import gc
import os # Import os

# Global variables for LLM clients
gemma_client = None
llama_client = None

# Global variable for RAG processor
rag_processor = None
DATA_DIR = "data"
INDEX_PATH = os.path.join(DATA_DIR, "faiss_index.idx") # Path relative to project root

# Ensure data directory exists
os.makedirs(DATA_DIR, exist_ok=True)

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

def chat_function(message: str, history: list, model_selection: str, rag_enabled: bool) -> str:
    """
    Core chat function called by Gradio.
    Handles model selection, RAG, history conversion, and response generation.
    Returns the bot's response string.
    """
    global gemma_client, llama_client, rag_processor

    if not model_selection:
        return "ERROR: Please select a model from the dropdown and click 'Load Selected Model'."

    # Convert Gradio history to the format expected by the LLM clients
    llm_history = convert_gradio_history_to_llm_format(history)
    
    retrieved_context = None
    if rag_enabled:
        if rag_processor and rag_processor.index is not None and rag_processor.document_chunks:
            print(f"RAG enabled. Retrieving context for: '{message}'")
            retrieved_context = rag_processor.retrieve_context(query=message)
            if "Error:" in retrieved_context or "RAG Index not available" in retrieved_context:
                 # If RAG retrieval had an error, pass it as part of the bot's response or log it.
                 # For now, we'll log it and proceed without context.
                 print(f"RAG Retrieval Info/Error: {retrieved_context}")
                 # Optionally, inform user: bot_response += f"\n[RAG Info: {retrieved_context}]"
                 retrieved_context = None # Do not pass error messages as context to LLM
            else:
                print(f"Retrieved context (first 100 chars): {retrieved_context[:100]}...")
        else:
            # RAG is enabled, but index is not ready.
            # We could return an error, or proceed without RAG.
            # For a smoother UX, proceeding without RAG and relying on LLM's general knowledge.
            # A status could be displayed elsewhere or a subtle notification.
            print("RAG is enabled, but the index is not ready or RAG processor not initialized. Proceeding without RAG.")
            # Optionally: return "ERROR: RAG is enabled, but index is not ready. Please build it or initialize RAG."

    bot_response = ""
    selected_client = None
    client_name = ""

    if model_selection == GEMMA_MODEL_NAME:
        selected_client = gemma_client
        client_name = GEMMA_MODEL_NAME
    elif model_selection == LLAMA_MODEL_NAME:
        selected_client = llama_client
        client_name = LLAMA_MODEL_NAME
    
    if selected_client is None:
        return f"ERROR: {client_name or 'Selected model'} is not loaded. Please load it first."

    try:
        print(f"Sending to {client_name}: '{message}' with history (len {len(llm_history)}) and RAG context (present: {bool(retrieved_context)})")
        bot_response = selected_client.generate_response(
            user_prompt=message, 
            chat_history=llm_history,
            retrieved_context=retrieved_context
        )
        print(f"{client_name} response: {bot_response}")
    
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

    # RAG processor and its index are kept loaded as they are not tied to a specific LLM's GPU memory
    # and are generally smaller / CPU bound for the index itself.
    # If rag_processor needed explicit cleanup, it would go here.
    # For example:
    # if rag_processor is not None:
    #     del rag_processor
    #     rag_processor = None
    #     status_message += "RAG processor reset. "


    if torch.cuda.is_available():
        print("Clearing CUDA cache...")
        torch.cuda.empty_cache()
        status_message += "CUDA cache cleared. "
        
    print("Running garbage collection...")
    gc.collect()
    status_message += "Garbage collection run."
    
    print(status_message)
    return [], status_message # Clears chatbot and updates status display


# --- RAG Functions ---
def initialize_rag_processor():
    """Initializes or re-initializes the RAGProcessor."""
    global rag_processor
    try:
        print("Initializing RAG Processor...")
        # Note: document_dir should point to where user's .txt/.md files for RAG are.
        # For this app, we assume they are placed in DATA_DIR by the user.
        rag_processor = RAGProcessor(document_dir=DATA_DIR, index_path=INDEX_PATH)
        # RAGProcessor's __init__ already attempts to load an existing index.
        if rag_processor.index is not None and rag_processor.document_chunks:
            return f"RAG Processor initialized. Existing index loaded with {len(rag_processor.document_chunks)} chunks."
        elif rag_processor.embedding_model is None:
             return "RAG Processor initialized, BUT EMBEDDING MODEL FAILED TO LOAD. RAG will not work."
        else:
            return "RAG Processor initialized. Index not found or empty. Build the index if you have documents in the 'data' directory."
    except Exception as e:
        rag_processor = None # Ensure it's None on failure
        error_msg = f"Error initializing RAG Processor: {str(e)}"
        print(error_msg)
        return error_msg

def build_rag_index_action():
    """Builds the RAG index using documents in the DATA_DIR."""
    global rag_processor
    if rag_processor is None:
        return "RAG Processor not initialized. Please initialize first."
    if rag_processor.embedding_model is None:
        return "Cannot build RAG index: Embedding model failed to load during RAG processor initialization."
    
    try:
        print("Building RAG index...")
        # RAGProcessor.build_index itself prints detailed status.
        # We can augment this with a summary here.
        rag_processor.build_index() # This method should handle printing its own status.
        
        if rag_processor.index is not None and rag_processor.document_chunks:
            return f"RAG index built/updated successfully with {len(rag_processor.document_chunks)} chunks."
        elif not rag_processor.document_chunks:
             return "RAG index building process completed, but no document chunks were found or loaded. Ensure '.txt' or '.md' files with content are in the 'data' directory."
        else:
            return "RAG index building process completed, but the index or chunks are still not available. Check logs."
    except Exception as e:
        error_msg = f"Error building RAG index: {str(e)}"
        print(error_msg)
        return error_msg


# --- Gradio Interface Definition ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("<h1>LLM Chatbot with RAG (Gemma 3 & Llama 3.2)</h1>")
    gr.Markdown(
        "Select an LLM, load it, and start chatting. "
        "Optionally, initialize the RAG processor, build an index from documents in the `data/` directory, "
        "and enable RAG to augment responses with retrieved context."
    )

    with gr.Tab("LLM and RAG Controls"):
        gr.Markdown("## LLM Selection and Loading")
        with gr.Row():
            model_selector = gr.Dropdown(
                choices=[GEMMA_MODEL_NAME, LLAMA_MODEL_NAME],
                label="Select LLM Model",
                info="Choose the LLM you want to interact with."
            )
            load_button = gr.Button("Load Selected LLM")
        status_display = gr.Textbox(label="LLM Status", interactive=False, placeholder="LLM loading status will appear here...")

        gr.Markdown("## RAG (Retrieval Augmented Generation) Controls")
        gr.Markdown(
            "Place your `.txt` or `.md` document files into the `data` directory in the project folder. "
            "Then, initialize the RAG processor and build the index."
        )
        rag_status_display = gr.Textbox(
            label="RAG Status", 
            interactive=False, 
            value="RAG Processor not initialized. Click 'Initialize RAG Processor'."
        )
        with gr.Row():
            initialize_rag_button = gr.Button("Initialize RAG Processor")
            build_index_button = gr.Button("Build/Update RAG Index from 'data/' dir")
        
        rag_checkbox = gr.Checkbox(label="Enable RAG for Chatbot Responses", value=False)

    with gr.Tab("Chat Interface"):
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

    # --- Event Handling ---
    load_button.click(
        fn=load_model,
        inputs=model_selector,
        outputs=status_display
    )

    initialize_rag_button.click(
        fn=initialize_rag_processor,
        outputs=rag_status_display
    )

    build_index_button.click(
        fn=build_rag_index_action,
        outputs=rag_status_display
    )

    # Connect send button
    send_button.click(
        fn=chat_function, 
        inputs=[msg_textbox, chatbot, model_selector, rag_checkbox], # Added rag_checkbox
        outputs=chatbot 
    ).then(lambda: "", outputs=[msg_textbox]) 

    # Connect textbox submit (pressing Enter)
    msg_textbox.submit(
        fn=chat_function,
        inputs=[msg_textbox, chatbot, model_selector, rag_checkbox], # Added rag_checkbox
        outputs=chatbot
    ).then(lambda: "", outputs=[msg_textbox])

    clear_button.click(
        fn=clear_chat_and_unload_models,
        inputs=None,
        outputs=[chatbot, status_display] # Clears chatbot and updates status
    )

if __name__ == "__main__":
    print("Launching Gradio app...")
    # Share=True for public link (optional, remove if not needed for local testing)
    # Set debug=True for more detailed error messages during development.
    
    # Initial RAG processor initialization is deferred to user click for clarity.
    # If automatic initialization on startup is desired:
    # initial_rag_status = initialize_rag_processor()
    # rag_status_display.value = initial_rag_status 
    # print(f"Initial RAG status: {initial_rag_status}")

    demo.launch(debug=True)
    # demo.launch() # For production or when debugging is not needed.
