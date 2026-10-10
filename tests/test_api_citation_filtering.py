from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.mark.parametrize(
    "endpoint",
    ["/chat", "/chat/multi"],
)
@pytest.mark.parametrize(
    "answer, expected_numbers",
    [
        ("Example [Source 1, page 133]", [1]),
        ("Example [Source 1, page 999]", []),
        ("Example without citations", []),
        (
            "Example [Source 1, page 133] "
            "[Source 2, page 134]",
            [1, 2],
        ),
    ],
)
def test_real_api_citation_filtering(
    endpoint,
    answer,
    expected_numbers,
    tmp_path,
):
    evidence = [
        {
            "chunk_id": "chunk_1",
            "document_id": "test_doc",
            "source_file": "test.pdf",
            "page_number": 133,
            "score": 0.9,
            "reranker_score": 0.9,
            "text": "First example.",
        },
        {
            "chunk_id": "chunk_2",
            "document_id": "test_doc",
            "source_file": "test.pdf",
            "page_number": 134,
            "score": 0.8,
            "reranker_score": 0.8,
            "text": "Second example.",
        },
    ]

    chunks = tmp_path / "chunks.json"
    index = tmp_path / "index.faiss"
    chunks.touch()
    index.touch()

    document = {
        "chunks_file_path": str(chunks),
        "faiss_index_path": str(index),
    }

    payload = {
        "question": "Give an example.",
        "top_k": 2,
    }

    if endpoint == "/chat":
        payload["document_id"] = "test_doc"
    else:
        payload["document_ids"] = ["test_doc", "test_doc_2"]

    with (
        patch(
            "app.api.chat.resolve_session",
            return_value="test_session",
        ),
        patch(
            "app.api.chat.get_recent_messages",
            return_value=[],
        ),
        patch(
            "app.api.chat.rewrite_query",
            return_value="Give an example.",
        ),
        patch(
            "app.api.chat.get_document",
            return_value=document,
        ),
        patch(
            "app.api.chat.retrieve_decomposed_queries",
            return_value=evidence,
        ),
        patch(
            "app.api.chat.build_diverse_multi_results",
            return_value=evidence,
        ),
        patch(
            "app.api.chat.select_llm_evidence",
            return_value=evidence,
        ),
        patch(
            "app.api.chat.generate_rag_answer",
            return_value=answer,
        ),
        patch("app.api.chat.add_message"),
    ):
        response = TestClient(app).post(
            endpoint,
            json=payload,
        )

    assert response.status_code == 200, response.text

    data = response.json()

    assert data["answer"] == answer

    actual_numbers = [
        source["source_number"]
        for source in data["sources"]
    ]

    assert actual_numbers == expected_numbers

    for source in data["sources"]:
        assert (
            f"[Source {source['source_number']}, "
            f"page {source['page_number']}]"
        ) in answer
