import streamlit as st
import requests
import os

# --- Configuration ---
# IMPORTANT: Make sure this URL is correct for your deployed Render service
API_URL = "https://rag-api-service-ajbermudezh22.onrender.com" 
PAGE_LIMIT = 200

# --- UI Setup ---
st.set_page_config(page_title="Chat with Your Document", layout="wide")
st.title("📄 Chat with Your Document")

# --- Session State Management ---
if "session_id" not in st.session_state:
    st.session_state.session_id = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Sidebar for Uploading ---
with st.sidebar:
    st.header("1. Upload Your Document")
    st.markdown(f"**Note:** Documents are limited to {PAGE_LIMIT} pages.")
    uploaded_file = st.file_uploader("Upload a PDF file to begin", type="pdf")
    
    if st.button("Process Document") and uploaded_file:
        with st.spinner("Processing document... This may take a few minutes depending on the size."):
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
            try:
                # The '/upload' endpoint is where the new backend logic lives
                response = requests.post(f"{API_URL}/upload", files=files, timeout=600) # 10 minute timeout for large docs
                if response.status_code == 200:
                    data = response.json()
                    # We store the unique session_id returned by the backend
                    st.session_state.session_id = data["session_id"]
                    st.session_state.messages = [] # Clear previous chat history
                    st.success("Document processed! You can now ask questions below.")
                else:
                    st.error(f"Error ({response.status_code}): {response.json().get('detail', 'Unknown processing error')}")
            except requests.RequestException as e:
                st.error(f"Connection error: Failed to reach the API. Please ensure the backend is running. Details: {e}")

# --- Main Chat Interface ---
st.header("2. Ask Questions")

if st.session_state.session_id:
    # Display previous messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Handle new user input
    if prompt := st.chat_input("Ask a question about your document"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get assistant's response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    payload = {"session_id": st.session_state.session_id, "question": prompt}
                    # The '/ask' endpoint now requires a session_id
                    response = requests.post(f"{API_URL}/ask", json=payload, timeout=120) # 2 minute timeout for answers
                    
                    if response.status_code == 200:
                        data = response.json()
                        answer = data["answer"]
                        st.markdown(answer)
                        with st.expander("Show Sources"):
                            st.json(data["sources"])
                        st.session_state.messages.append({"role": "assistant", "content": answer})
                    else:
                        st.error(f"Error ({response.status_code}): {response.json().get('detail', 'Failed to get answer')}")
                except requests.RequestException as e:
                    st.error(f"Connection error: {e}")
else:
    st.info("Please upload and process a document in the sidebar to begin chatting.")