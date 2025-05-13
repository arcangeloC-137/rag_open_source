import streamlit as st

import streamlit as st
import ollama
import subprocess
import json

# Function to get available models from Ollama
def get_local_models():
    try:
        output = subprocess.run(["ollama", "list"], capture_output=True, text=True)
        models = [line.split()[0] for line in output.stdout.splitlines()[1:]]  # Skip header
        return models
    except Exception as e:
        st.error(f"Error fetching models: {e}")
        return []

# Function to download a new model
def download_model(model_name):
    try:
        with st.spinner(f"Downloading model '{model_name}'..."):
            subprocess.run(["ollama", "pull", model_name], check=True)
        st.success(f"Model '{model_name}' downloaded successfully!")
        st.rerun()  # Refresh UI to show the new model
    except subprocess.CalledProcessError as e:
        st.error(f"Failed to download model '{model_name}': {e}")

# Streamlit UI
st.set_page_config(page_title="Ollama Chatbot", page_icon="💬", layout="wide")
st.title("💬 Simple Chatbot with Ollama API")

# Sidebar for model selection
with st.sidebar:
    st.header("Model Selection")
    available_models = get_local_models()
    
    if available_models:
        model_choice = st.selectbox("Choose a model", available_models)
    else:
        st.warning("No models found. Please download one.")
        model_choice = None

    new_model = st.text_input("Download a new model (e.g., mistral, llama3)")
    if st.button("Download Model") and new_model:
        download_model(new_model)

# Chat history
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Display chat history
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User input
if model_choice:
    user_input = st.chat_input("Type your message...")
    
    if user_input:
        # Append user message
        st.session_state["messages"].append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Prepare chat history for Ollama API
        ollama_messages = [{"role": msg["role"], "content": msg["content"]} for msg in st.session_state["messages"]]
        
        # Get response from Ollama
        try:
            with st.spinner():
                response = ollama.chat(model=model_choice, messages=ollama_messages)
                bot_reply = response["message"]["content"].split("</think>")[-1]
        except Exception as e:
            bot_reply = f"Error: {e}"

        # Append bot message
        st.session_state["messages"].append({"role": "assistant", "content": bot_reply})
        with st.chat_message("assistant"):
            st.markdown(bot_reply)
else:
    st.info("Please select or download a model to start chatting.")
