import sys
import os
import json
import time
import random
import datetime
import pandas as pd

import streamlit as st
import streamlit.components.v1 as components

import logging
logging.basicConfig(level=logging.INFO)

main_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, main_dir)

# Import relevant modules
from databricks_scripts.multi_chain import MultiStageSystemWrapper
from common.utils import print_nested_dict_display

from common.databricks_config import (
    DATABRICKS_HOST, 
    DATABRICKS_TOKEN, 
    DATABRICKS_WAREHOUSE_ID,
    catalog_name, 
    schema_name
)

# Dynamically get the path to the assets directory and chat history file
current_dir = os.path.dirname(os.path.abspath(__file__))
svg_logo_path = os.path.join(current_dir, "assets", "logo-regubim-ai.svg")
ico_logo_path = os.path.join(current_dir, "assets", "logo-regubim-ai.ico")
chat_history_path = os.path.join(current_dir, "chat_history.json")
visualisation_template_path = os.path.join(current_dir, "visualisation_templates", "room-route-visualisation-min.html")
visualisation_iframe_height = 500

# Set a new icon
st.set_page_config(
    page_title="ReguBIM AI",
    page_icon=ico_logo_path
)

# Load Font Awesome CDN
st.markdown("""<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.4/css/all.min.css">""", unsafe_allow_html=True)


if not os.path.exists(visualisation_template_path):
    logging.error(f"Visualization template not found at: {visualisation_template_path}")
    

# Load SVG logo (ensure the correct path)
if os.path.exists(svg_logo_path):
    with open(svg_logo_path, "r") as svg_file:
        svg_logo = svg_file.read()
else:
    st.error("SVG logo file not found")


# List of loading messages
loading_messages = [
    "ReguBIM AI is processing your request..."
]

def display_welcome_message():
    st.markdown("""
    <div style="display: flex; justify-content: center; align-items: center; height: 70vh;">
        <div style="text-align: center; padding: 20px; background-color: #f0f2f6; border-radius: 10px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);">
            <h2>Welcome to ReguBIM AI!</h2>
            <p>How can I assist you today?</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Function to load chat history
def load_chat_history():
    if os.path.exists(chat_history_path):
        with open(chat_history_path, "r") as f:
            return json.load(f)
    return []

# Function to save chat history
def save_chat_history(history):
    with open(chat_history_path, "w") as f:
        json.dump(history, f, default=lambda o: '<not serializable>')

def call_databricks_chain(prompt, max_time=180, poll_interval=10):
    try:
        print(f"Processing query: {prompt}")
        
        query_df = pd.DataFrame([{"query": prompt, "debug_mode": False}])
        
        llm_model = "databricks-meta-llama-3-1-70b-instruct"
        multi_chain_wrapper = MultiStageSystemWrapper(llm_model, catalog_name, schema_name, DATABRICKS_HOST, DATABRICKS_TOKEN, DATABRICKS_WAREHOUSE_ID)
        
        start_time = time.time()
        warning_issued = False
        
        while True:
            result = multi_chain_wrapper.predict(query_df)
            
            if result is not None:
                elapsed_time = time.time() - start_time
                print(f"Waiting... Elapsed time: {elapsed_time:.1f} seconds")
                try:
                    return result[0]
                except json.JSONDecodeError:
                    return {"error": f"Error: {str(e)}"}
    
    except Exception as e:
        return {"error": f"Error: {str(e)}"}

    
def print_output_and_visualisation(data):
    def format_value(value):
        if isinstance(value, dict):
            return f"```json\n{json.dumps(value, indent=2)}\n```"
        elif isinstance(value, list):
            return "\n".join([f"- {item}" for item in value])
        elif isinstance(value, str):
            try:
                json_data = json.loads(value)
                return f"```json\n{json.dumps(json_data, indent=2)}\n```"
            except json.JSONDecodeError:
                return value
        else:
            return str(value)

    output_md = []
    html_visualisation = None

    # Handle 'output' key
    if 'output' in data and data['output'] not in [None, 'null', 'None']:
        output_md.append(f"{format_value(data['output'])}")

    # Handle 'bim-revit-data-chain-output'
    if 'bim-revit-data-chain-output' in data and data['bim-revit-data-chain-output'] not in [None, 'null', 'None']:
        output_md.append(f"{format_value(data['bim-revit-data-chain-output'])}")

        # Check for 'bim-revit-data-visualisation-json-output' and set HTML if present
        visualisation_data = data.get('bim-revit-data-visualisation-json-output')
        if visualisation_data not in [None, 'null', 'None']:
            output_md.append("**View the route visualization graph below to explore all possible routes:**")
            html_visualisation = load_visualisation_html(visualisation_data, "bim")

    # Handle 'compliance_check_chain_output'
    if 'compliance_check_chain_output' in data and data['compliance_check_chain_output'] not in [None, 'null', 'None']:
        output_md.append(f"{format_value(data['compliance_check_chain_output'])}")

        # Check for 'bim-revit-data-visualisation-json-output' and set HTML if present
        visualisation_data = data.get('bim_revit_data_visualisation_json_output')
        if visualisation_data not in [None, 'null', 'None']:
            output_md.append("**View the route visualization graph below to explore all possible routes:**")
            html_visualisation = load_visualisation_html(visualisation_data, "compliance")

    # Return both the markdown content and HTML visualization
    return "\n\n".join(output_md), html_visualisation

# Function to load and replace the placeholder in the HTML
def load_visualisation_html(visualisation_json, type_of_output):
    try:
        # Load the HTML template from file
        with open(visualisation_template_path, "r") as html_file:
            html_content = html_file.read()
            
        if type_of_output == "compliance":
            # Replace the placeholder 'path_graph_json' with the JSON data
            modified_html_content = html_content.replace("'''path_graph_json'''", visualisation_json)
        else:
            # Replace the placeholder 'path_graph_json' with the JSON data
            modified_html_content = html_content.replace("'''path_graph_json'''", json.dumps(visualisation_json))

        # Return the HTML content
        return modified_html_content
    except Exception as e:
        return f"Error loading visualisation: {e}"
    
# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = load_chat_history()
if 'current_chat_index' not in st.session_state:
    st.session_state.current_chat_index = None
if 'loading' not in st.session_state:
    st.session_state.loading = False
if 'loading_message' not in st.session_state:
    st.session_state.loading_message = random.choice(loading_messages)

# Custom CSS for the spinner and layout
st.markdown("""
    <style>

        div[data-testid="stDecoration"] {
                background-image: linear-gradient(90deg, #00d2ff, #3a7bd5) !important;
        }

        /* Chat message styling */
        .chat-message {
            display: flex;
            align-items: flex-start;
            padding: 1rem;
        }

        .chat-message.assistant {
            background-color: #f0f2f6;
        }

        .chat-message .avatar {
            width: 40px;
            height: 40px;
            border-radius: 50%;
            object-fit: cover;
            margin-right: 1rem;
        }

        .chat-message .message {
            flex: 1;
        }

        /* Spinner animation */
        .spinner {
            width: 20px;
            height: 20px;
            border-radius: 50%;
            background: conic-gradient(from 0deg, #00d2ff, #3a7bd5);
            mask-image: radial-gradient(farthest-side, transparent calc(100% - 2px), black calc(100% - 1px));
            animation: spin 1s linear infinite;
            display: inline-block;
            margin-right: 10px;
            vertical-align: middle;
        }


        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }

        /* Loading text animation */
        .loading-container {
            display: flex;
            align-items: center;
            margin-left: 5px;
            margin-top: -10px;
        }

        .loading-text {
            display: inline-block;
            vertical-align: middle;
            margin-left: 5px;
            color: gray;
            animation: ripple 2s ease-in-out infinite;
        }

        @keyframes ripple {
            0% { transform: translateY(0); }
            25% { transform: translateY(-2px); }
            50% { transform: translateY(0); }
            75% { transform: translateY(2px); }
            100% { transform: translateY(0); }
        }

        /* Sidebar customization */
        .sidebar .sidebar-content {
            background-color: #f0f0f5;
            padding: 10px;
        }

        /* Button styling */
        .stButton > button {
            font-size: 12px !important;
            border-radius: 5px !important;
            padding: 8px 10px !important;
            border: none !important;
            transition: all 0.3s ease !important;
            width: 100% !important;
            cursor: pointer !important;
            margin-bottom: 5px !important;
        }

        .stButton > button:hover {
            background-color: #007bff !important;  /* Blue background on hover */
            color: white !important;  /* White text on hover */
        }

        .stButton > button:active,
        .stButton > button:focus {
            background-color: #0056b3 !important;  /* Darker blue for active/selected state */
            color: white !important;  /* White text for active/selected state */
            outline: none !important;  /* Remove default focus outline */
            box-shadow: 0 0 0 2px rgba(0,123,255,0.5) !important;  /* Add a subtle blue glow */
        }

        /* Focus effects for input fields */
        [data-baseweb="textarea"] textarea {
            border-color: #ccc;
            transition: border-color 0.3s, box-shadow 0.3s;
        }

        [data-baseweb="textarea"] textarea:focus {
            border-color: #007BFF !important;
            box-shadow: 0 0 0 0.2rem rgba(0, 123, 255, 0.25) !important;
        }

        /* Style for the textarea container */
        [data-baseweb="textarea"] {
            border-radius: 0.375rem;
        }

        [data-baseweb="textarea"]:focus-within {
            border-color: #007BFF !important;
            box-shadow: 0 0 0 0.2rem rgba(0, 123, 255, 0.25) !important;
        }

        .stTextInput > div > div > input:focus {
            border-color: #007BFF !important;
            box-shadow: 0 0 0 0.2rem rgba(0, 123, 255, 0.25) !important;
        }

        [data-testid="stChatMessageAvatarAssistant"]{
            background-color: #007BFF;
        }

        [data-testid="stChatMessageAvatarUser"]{
            background-color: #ffc107;
        }

    </style>
""", unsafe_allow_html=True)

# Sidebar for past chats and settings
st.sidebar.markdown(f'''
    <div style="text-align: center; margin-bottom: 0px;">
        <svg width="100%" height="100" viewBox="0 0 1110 255" xmlns="http://www.w3.org/2000/svg">
            {svg_logo}
        </svg>
    </div>
''', unsafe_allow_html=True)
st.sidebar.title("Chat History")

# New Chat and Clear All Chats buttons side by side
with st.sidebar.container():
    col1, col2 = st.columns(2)
    with col1:
        new_chat = st.button("New Chat", key="new_chat")
    with col2:
        clear_all = st.button("Clear All Chats", key="clear_all")

    if new_chat:
        st.session_state.messages = []
        st.session_state.current_chat_index = None
        st.rerun()
        
    if clear_all:
        st.session_state.chat_history = []
        st.session_state.messages = []
        st.session_state.current_chat_index = None
        save_chat_history(st.session_state.chat_history)
        st.rerun()

# Display past chats in sidebar
for i, chat in enumerate(st.session_state.chat_history):
    col1, col2 = st.sidebar.columns([4, 1])
    with col1:
        if st.button(f"Chat {i+1}: {chat['timestamp']}", key=f"chat_{i}"):
            st.session_state.current_chat_index = i
            st.session_state.messages = chat['messages']
            st.rerun()
    with col2:
        if st.button("🗑️", key=f"delete_{i}"):
            del st.session_state.chat_history[i]
            if st.session_state.current_chat_index is not None:
                if st.session_state.current_chat_index == i:
                    st.session_state.current_chat_index = None
                    st.session_state.messages = []
                elif st.session_state.current_chat_index > i:
                    st.session_state.current_chat_index -= 1
            save_chat_history(st.session_state.chat_history)
            st.rerun()

# Chat input for user query
if prompt := st.chat_input("Ask a question:"):
    if st.session_state.current_chat_index is None:
        # Start a new chat
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.session_state.chat_history.append({
            "timestamp": timestamp,
            "messages": []
        })
        st.session_state.current_chat_index = len(st.session_state.chat_history) - 1

    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.chat_history[st.session_state.current_chat_index]["messages"] = st.session_state.messages
    st.session_state.loading = True
    save_chat_history(st.session_state.chat_history)
    st.rerun()

# Modify the part where we display past messages to include visualizations
if st.session_state.messages:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and "visualization" in message and message["visualization"]:
                components.html(message["visualization"], height=visualisation_iframe_height, scrolling=True)
else:
    display_welcome_message()
    
# Handle loading state and model response
if st.session_state.loading:
    # Display loading message outside of chat messages
    loading_placeholder = st.empty()
    loading_placeholder.markdown(f"""
    <div class="loading-container">
        <div class="spinner"></div>
        <div class="loading-text">{st.session_state.loading_message}</div>
    </div>
    """, unsafe_allow_html=True)

    try:
        chain_response = call_databricks_chain(prompt=st.session_state.messages[-1]["content"])
        
        if chain_response:
            st.session_state.loading = False
            loading_placeholder.empty()  # Remove the loading message
        
            try:
                response_dict = json.loads(chain_response)
                if "error" in response_dict:
                    st.error(f"An error occurred: {response_dict['error']}")
                else:
                    # Process and display the response as before
                    markdown_content, html_visualisation = print_output_and_visualisation(response_dict)
                    
                    with st.chat_message("assistant"):
                        st.markdown(markdown_content)
                        if html_visualisation:
                            components.html(html_visualisation, height=visualisation_iframe_height, scrolling=True)

                    # Update chat history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": markdown_content,
                        "visualization": html_visualisation if html_visualisation else None
                    })

                    st.session_state.chat_history[st.session_state.current_chat_index]["messages"] = st.session_state.messages
                    save_chat_history(st.session_state.chat_history)
            except json.JSONDecodeError:
                st.error("Received an invalid response from the model.")
    except Exception as e:
        st.session_state.loading = False
        loading_placeholder.empty()  # Remove the loading message
        st.error(f"An error occurred while processing your request: {str(e)}")

    st.rerun()

# Update loading message
if st.session_state.loading:
    st.session_state.loading_message = random.choice(loading_messages)