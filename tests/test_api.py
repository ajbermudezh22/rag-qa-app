import pytest
from fastapi.testclient import TestClient
from main import app, get_rag_chain, get_retriever, get_rag_initializer

# --- Mock Implementations ---

def mock_rag_chain_invoke(query: str):
    """Mock implementation for the RAG chain's invoke method."""
    return "This is a mock answer about Paris."

def mock_retriever_invoke(query: str):
    """Mock implementation for the retriever's invoke method."""
    class MockDoc:
        def __init__(self, page_content, metadata):
            self.page_content = page_content
            self.metadata = metadata
    return [
        MockDoc(page_content="Paris is the capital of France.", metadata={"source": "mock_source_1"}),
    ]

# This is a mock for the chain and retriever objects themselves
class MockRunnable:
    def invoke(self, *args, **kwargs):
        pass

mock_rag_chain_instance = MockRunnable()
mock_rag_chain_instance.invoke = mock_rag_chain_invoke

mock_retriever_instance = MockRunnable()
mock_retriever_instance.invoke = mock_retriever_invoke

def mock_rag_initializer():
    """A mock for the initialize_rag_chain function."""
    print("Called mock_rag_initializer")
    return mock_rag_chain_instance, mock_retriever_instance

# --- Override Dependencies BEFORE creating the TestClient ---

# This is the key fix: We override the initializer dependency.
# When the TestClient starts the app, the lifespan function will call our mock_rag_initializer
# instead of the real one, preventing the crash.
app.dependency_overrides[get_rag_initializer] = lambda: mock_rag_initializer

# We also still need to override the dependencies for the endpoint itself.
app.dependency_overrides[get_rag_chain] = lambda: mock_rag_chain_instance
app.dependency_overrides[get_retriever] = lambda: mock_retriever_instance

# --- Create the Test Client ---
# This must be done AFTER the overrides are in place.
client = TestClient(app)


# --- Tests ---

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

def test_ask_question_success():
    response = client.post("/ask", json={"question": "What is the capital of France?"})
    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "This is a mock answer about Paris."
    assert len(data["sources"]) == 1
    assert data["sources"][0]["source"] == "mock_source_1"

def test_ask_question_empty():
    response = client.post("/ask", json={"question": "  "})
    assert response.status_code == 400
    assert response.json() == {"detail": "Question cannot be empty"}

def test_ask_question_no_body():
    response = client.post("/ask")
    assert response.status_code == 422
