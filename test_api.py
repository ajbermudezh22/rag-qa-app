import pytest
from fastapi.testclient import TestClient
from main import app 

# Create a TestClient instance
client = TestClient(app)

def test_health_check():
    """
    Tests the /health endpoint to ensure the API is running.
    """
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

def test_ask_question_success():
    """
    Tests the /ask endpoint with a valid question.
    """
    # This is a simple test case. It doesn't check the answer's content,
    # just that the endpoint returns a successful response in the correct format.
    response = client.post("/ask", json={"question": "What is the capital of France?"})
    
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "sources" in data
    assert isinstance(data["answer"], str)
    assert isinstance(data["sources"], list)

def test_ask_question_empty():
    """
    Tests the /ask endpoint with an empty question to ensure it returns an error.
    """
    response = client.post("/ask", json={"question": "  "})
    assert response.status_code == 400  # Bad Request
    assert response.json() == {"detail": "Question cannot be empty"}

def test_ask_question_no_body():
    """
    Tests calling the /ask endpoint without a request body.
    """
    response = client.post("/ask")
    assert response.status_code == 422  # Unprocessable Entity (FastAPI's validation error)