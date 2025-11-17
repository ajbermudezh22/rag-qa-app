# End-to-End RAG Q&A Application: "Chat with Your Document"

**A live demo of this application is available here:** [https://rag-app-app-ofwphtytyhezmepeo8kqnc.streamlit.app/](https://rag-app-app-ofwphtytyhezmepeo8kqnc.streamlit.app/)

This repository contains the source code for a full-stack, AI-powered web application that allows users to upload a PDF document and ask questions about its content. This project was built from the ground up as a production-level demonstration, encompassing everything from local development to a live, public deployment.

The application leverages a Retrieval-Augmented Generation (RAG) architecture to provide accurate, context-aware answers based solely on the content of the uploaded document.

## Features

-   **Dynamic Document Upload**: Users can upload any PDF document (up to 200 pages).
-   **Interactive Chat Interface**: A user-friendly, session-based chat interface to "talk" to the document.
-   **Contextual Answers**: The AI generates answers based on relevant passages retrieved from the document.
-   **Source Attribution**: The application shows which parts of the document were used to generate an answer, providing transparency.
-   **Scalable Architecture**: A decoupled frontend and backend architecture built for scalability and maintainability.

## Technical Architecture

This project is a demonstration of a modern, end-to-end software development lifecycle, utilizing a robust stack of technologies and professional practices.

### 1. Frontend (User Interface)

-   **Framework**: [Streamlit](https://streamlit.io/)
-   **Role**: Provides the user interface for document upload and interactive chat. It is a purely client-facing application.
-   **Deployment**: Deployed on **Streamlit Community Cloud**, which automatically redeploys on every push to the `main` branch.

### 2. Backend (API & AI Logic)

-   **Framework**: [FastAPI](https://fastapi.tiangolo.com/)
-   **Role**: A powerful, containerized REST API that handles all the heavy lifting:
    -   Receives PDF uploads and creates session-based RAG chains.
    -   Manages in-memory vector stores for each user session using `FAISS`.
    -   Processes user questions, performs semantic search, and generates answers using the OpenAI API.
-   **Containerization**: The entire backend is containerized using **Docker**, ensuring consistency across development and production environments.
-   **Deployment**: The Docker image is hosted on **Docker Hub** and deployed as a Web Service on **Render.com**.

### 3. AI Core (RAG Pipeline)

-   **Framework**: [LangChain](https://www.langchain.com/)
-   **Embeddings**: `OpenAI text-embedding-3-small` is used to convert text chunks into vector representations.
-   **Vector Store**: `FAISS` (Facebook AI Similarity Search) is used to create a fast, in-memory vector index for each user session, allowing for efficient semantic search.
-   **LLM**: `OpenAI gpt-3.5-turbo` is used for generating the final, human-readable answers based on the retrieved context.

### 4. Professional Practices & DevOps

-   **Version Control**: All code is managed in a **Git** repository hosted on **GitHub**.
-   **Testing**: The FastAPI backend includes a suite of unit tests written with **pytest**. These tests use mocking to ensure they can run without real API keys, making them suitable for automated environments.
-   **Continuous Integration (CI)**: A **GitHub Actions** workflow is configured to automatically run the test suite on every push and pull request to the `main` branch, ensuring code quality and preventing regressions.

## How to Run Locally

To set up and run this project on your local machine, follow these steps:

**Prerequisites**:
*   Python 3.10+
*   Docker
*   An OpenAI API Key

**Setup**:

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/ajbermudezh22/RAG-QA-app.git
    cd RAG-QA-app
    ```

2.  **Create a virtual environment**:
    ```bash
    python -m venv .venv
    source .venv/bin/activate
    ```

3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up environment variables**:
    Create a `.env` file in the project root and add your API key:
    ```
    OPENAI_API_KEY="sk-..."
    ```

**Running the Application**:

1.  **Start the backend server**:
    ```bash
    uvicorn main:app --reload
    ```
    The API will be available at `http://localhost:8000`.

2.  **Start the frontend UI** (in a new terminal):
    ```bash
    streamlit run ui.py
    ```
    The web application will open in your browser at `http://localhost:8501`. For the local UI to work, you will need to manually change the `API_URL` variable in `ui.py` to `http://localhost:8000`.

## Conclusion

This project is more than just an AI application; it is a complete, end-to-end solution that demonstrates skills in software architecture, API design, containerization, automated testing, CI/CD, and cloud deployment. It represents a full development cycle from concept to a live, scalable product.
