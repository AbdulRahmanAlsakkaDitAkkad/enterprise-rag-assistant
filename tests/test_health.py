from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_health_returns_200():
    response = client.get("/health")

    assert response.status_code == 200


def test_health_status_is_ok():
    response = client.get("/health")
    data = response.json()

    assert data["status"] == "ok"


def test_health_identifies_application():
    response = client.get("/health")
    data = response.json()

    assert data["message"] == (
        "Enterprise RAG Assistant API is running"
    )
