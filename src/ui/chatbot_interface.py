# Install Streamlit and Databricks libraries
# pip install streamlit databricks-sdk
# pip install databricks-sql-connector
# pip install streamlit openai
# streamlit run chatbot_interface.py

import streamlit as st
import databricks.sql as dbsql  # Databricks SQL for queries
import requests
import pandas as pd
from openai import OpenAI


# Set up Databricks connection details
#DATABRICKS_SERVER_HOSTNAME = "dbc-f8eeb0cc-8828.cloud.databricks.com"
#HTTP_PATH = "/sql/1.0/warehouses/0359c553d2f6fc02"
#DATABRICKS_TOKEN = "dapi4b78f22822537db0cfdf5a63236ca5e5"
#
#client = OpenAI(
#    api_key=DATABRICKS_TOKEN,
#    base_url="https://dbc-f8eeb0cc-8828.cloud.databricks.com/serving-endpoints"
#    
#)

DATABRICKS_SERVER_HOSTNAME = "dbc-f8eeb0cc-8828.cloud.databricks.com"
HTTP_PATH = "/sql/1.0/warehouses/0359c553d2f6fc02"
DATABRICKS_TOKEN = "dapi4b78f22822537db0cfdf5a63236ca5e5"

client = OpenAI(
    api_key=DATABRICKS_TOKEN,
    base_url="https://dbc-f8eeb0cc-8828.cloud.databricks.com/serving-endpoints"
    
)

# Streamlit UI Elements
# Load and display SVG logo using HTML, with custom height
svg_logo = open("assets/logo-augment.svg", "r").read()  # Adjust the path if needed


st.title("Databricks Chatbot")

# Sidebar for past chats and settings
# Add SVG logo to sidebar with custom width/height
st.sidebar.markdown(f'''
    <div style="text-align: center; margin-bottom: 0px;">
        <svg width="100%" height="100" viewBox="0 0 500 500" xmlns="http://www.w3.org/2000/svg">
            {svg_logo}
        </svg>
    </div>
    ''', unsafe_allow_html=True)
st.sidebar.title("Chat Settings")
st.sidebar.subheader("Past Chats")

# Initialize session state for chat messages
if "messages" not in st.session_state:
    st.session_state.messages = []
    
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # List to store past chats

# Add past chat history to the sidebar
for i, past_chat in enumerate(st.session_state.chat_history):
    with st.sidebar.expander(f"Chat {i + 1}"):
        for msg in past_chat:
            st.write(f"{msg['role']}: {msg['content']}")

# Function to call the Databricks model
def call_databricks_model(prompt):
    try:
        
        chat_completion = client.chat.completions.create(
        messages=[
        {
          "role": "system",
          "content": "You are an AI assistant",
        },
        {
          "role": "user",
          "content":prompt,
        }
        ],
        model="testing-llm-model",
        max_tokens=256
        )
        return chat_completion.choices[0].message.content  # Return the generated text
    
    except Exception as e:
        return f"Error: {str(e)}"

# Chat input for user query
if prompt := st.chat_input("Ask a question or request data:"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message in chat UI
    with st.chat_message("user"):
        st.markdown(prompt)

    # Call the Databricks model with the user input
    model_response = call_databricks_model(prompt)
    
    # Display model response in chat UI
    st.session_state.messages.append({"role": "assistant", "content": model_response})
    with st.chat_message("assistant"):
        st.markdown(model_response)

    # Save the current chat session to history
    st.session_state.chat_history.append(st.session_state.messages.copy())

    # Clear the current chat after sending a response
    st.session_state.messages = []  # Reset for the next interaction

# Optionally style the sidebar with custom CSS
st.markdown("""
    <style>
    .sidebar .sidebar-content {
        background-color: #f0f0f5;
        padding: 20px;
    }
    .stButton>button {
        color: white;
        background-color: #007BFF;
        border-radius: 5px;
        padding: 8px 20px;
    }
    </style>
    """, unsafe_allow_html=True)