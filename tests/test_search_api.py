from unittest.mock import patch

from fastapi.testclient import TestClient

from app.main import app


def test_semantic_search_success(tmp_path):
    chunks_path = tmp_path / "chunks.json"
    index_path = tmp_path / "index.faiss"

    chunks_path.touch()
    index_path.touch()

    results = [
        {
            "chunk_id": "chunk_1",
            "text": "Example relevant content.",
            "score": 0.95,
        }
    ]

    with patch(
        "app.api.search.search_faiss_index",
        return_value=results,
    ) as mocked_search:
        response = TestClient(app).post(
            "/search/semantic",
            params={
                "query": "Example question",
                "chunks_file_path": str(chunks_path),
                "faiss_index_path": str(index_path),
                "top_k": 3,
            },
        )

    assert response.status_code == 200, response.text

    assert response.json() == {
        "message": "Semantic search completed successfully",
        "query": "Example question",
        "top_k": 3,
        "results": results,
    }

    mocked_search.assert_called_once_with(
        query="Example question",
        index_path=index_path,
        chunks_file_path=chunks_path,
        top_k=3,
    )


def test_semantic_search_missing_chunks_file(tmp_path):
    chunks_path = tmp_path / "missing_chunks.json"
    index_path = tmp_path / "index.faiss"

    index_path.touch()

    with patch(
        "app.api.search.search_faiss_index",
    ) as mocked_search:
        response = TestClient(app).post(
            "/search/semantic",
            params={
                "query": "Example question",
                "chunks_file_path": str(chunks_path),
                "faiss_index_path": str(index_path),
            },
        )

    assert response.status_code == 404
    assert response.json() == {
        "detail": "Chunks file not found."
    }

    mocked_search.assert_not_called()


def test_semantic_search_missing_faiss_index(tmp_path):
    chunks_path = tmp_path / "chunks.json"
    index_path = tmp_path / "missing_index.faiss"

    chunks_path.touch()

    with patch(
        "app.api.search.search_faiss_index",
    ) as mocked_search:
        response = TestClient(app).post(
            "/search/semantic",
            params={
                "query": "Example question",
                "chunks_file_path": str(chunks_path),
                "faiss_index_path": str(index_path),
            },
        )

    assert response.status_code == 404
    assert response.json() == {
        "detail": "FAISS index file not found."
    }

    mocked_search.assert_not_called()


def test_semantic_search_internal_error(tmp_path):
    chunks_path = tmp_path / "chunks.json"
    index_path = tmp_path / "index.faiss"

    chunks_path.touch()
    index_path.touch()

    with patch(
        "app.api.search.search_faiss_index",
        side_effect=RuntimeError("FAISS search unavailable"),
    ) as mocked_search:
        response = TestClient(app).post(
            "/search/semantic",
            params={
                "query": "Example question",
                "chunks_file_path": str(chunks_path),
                "faiss_index_path": str(index_path),
            },
        )

    assert response.status_code == 500
    assert response.json() == {
        "detail": (
            "Semantic search failed: "
            "FAISS search unavailable"
        )
    }

    mocked_search.assert_called_once()
