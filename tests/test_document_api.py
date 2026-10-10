from unittest.mock import patch

from fastapi.testclient import TestClient

from app.main import app


def test_get_existing_document():
    document = {
        "document_id": "doc_test123",
        "original_file_name": "example.pdf",
        "extracted_file_path": "data/extracted/example.json",
        "chunks_file_path": "data/extracted/example_chunks.json",
        "embeddings_file_path": "data/indexes/example.npy",
        "faiss_index_path": "data/indexes/example.faiss",
    }

    with patch(
        "app.api.documents.get_document",
        return_value=document,
    ) as mocked_get:
        response = TestClient(app).get(
            "/documents/doc_test123"
        )

    assert response.status_code == 200
    assert response.json() == document
    mocked_get.assert_called_once_with("doc_test123")


def test_get_missing_document():
    with patch(
        "app.api.documents.get_document",
        return_value=None,
    ) as mocked_get:
        response = TestClient(app).get(
            "/documents/doc_missing"
        )

    assert response.status_code == 404
    assert response.json() == {
        "detail": "Document not found."
    }
    mocked_get.assert_called_once_with("doc_missing")


def test_upload_pdf_success():
    from pathlib import Path

    pages = [
        {"text": "First page of the document."},
        {"text": "   "},
        {"text": "Third page contains useful information."},
    ]

    with (
        patch(
            "app.api.documents.save_uploaded_pdf",
            return_value=Path("data/uploads/example.pdf"),
        ) as mocked_save,
        patch(
            "app.api.documents.extract_text_by_page",
            return_value=pages,
        ) as mocked_extract,
        patch(
            "app.api.documents.save_extracted_text",
            return_value=Path("data/extracted/example.json"),
        ) as mocked_extracted_save,
    ):
        response = TestClient(app).post(
            "/documents/upload",
            files={
                "file": (
                    "example.pdf",
                    b"%PDF-1.4 test content",
                    "application/pdf",
                )
            },
        )

    assert response.status_code == 200, response.text

    assert response.json() == {
        "message": "PDF uploaded and text extracted successfully",
        "file_name": "example.pdf",
        "pages_count": 3,
        "pages_with_text": 2,
        "extracted_file": str(Path("data/extracted/example.json")),
        "preview": "First page of the document.",
    }

    mocked_save.assert_called_once()
    mocked_extract.assert_called_once_with(
        Path("data/uploads/example.pdf")
    )
    mocked_extracted_save.assert_called_once_with(
        Path("data/uploads/example.pdf"),
        pages,
    )


def test_upload_pdf_invalid_file():
    with patch(
        "app.api.documents.save_uploaded_pdf",
        side_effect=ValueError("Only PDF files are allowed."),
    ) as mocked_save:
        response = TestClient(app).post(
            "/documents/upload",
            files={
                "file": (
                    "example.txt",
                    b"This is not a PDF.",
                    "text/plain",
                )
            },
        )

    assert response.status_code == 400
    assert response.json() == {
        "detail": "Only PDF files are allowed."
    }
    mocked_save.assert_called_once()


def test_upload_pdf_missing_file():
    response = TestClient(app).post(
        "/documents/upload"
    )

    assert response.status_code == 422

    errors = response.json()["detail"]

    assert any(
        error["loc"][-1] == "file"
        and error["type"] == "missing"
        for error in errors
    )


def test_process_document_success():
    from pathlib import Path
    from unittest.mock import Mock

    pages = [{"text": "Example document content."}]
    chunks = [{"text": "Example document content."}]

    embeddings = Mock()
    embeddings.shape = (1, 384)

    with (
        patch(
            "app.api.documents.save_uploaded_pdf",
            return_value=Path("data/uploads/example.pdf"),
        ),
        patch(
            "app.api.documents.extract_text_by_page",
            return_value=pages,
        ),
        patch(
            "app.api.documents.save_extracted_text",
            return_value=Path("data/extracted/example.json"),
        ),
        patch(
            "app.api.documents.create_chunks_from_extracted_json",
            return_value=chunks,
        ),
        patch(
            "app.api.documents.save_chunks",
            return_value=Path("data/extracted/chunks.json"),
        ),
        patch(
            "app.api.documents.create_embeddings_for_chunks",
            return_value=embeddings,
        ),
        patch(
            "app.api.documents.save_embeddings",
            return_value=Path("data/indexes/embeddings.npy"),
        ),
        patch(
            "app.api.documents.create_faiss_index",
            return_value=Mock(),
        ),
        patch(
            "app.api.documents.save_faiss_index",
            return_value=Path("data/indexes/index.faiss"),
        ),
        patch(
            "app.api.documents.register_document",
            return_value="doc_test123",
        ) as mocked_register,
    ):
        response = TestClient(app).post(
            "/documents/process",
            files={
                "file": (
                    "example.pdf",
                    b"%PDF-1.4 test content",
                    "application/pdf",
                )
            },
        )

    assert response.status_code == 200, response.text
    assert response.json() == {
        "message": "Document processed successfully",
        "document_id": "doc_test123",
        "original_file_name": "example.pdf",
        "pages_count": 1,
        "chunks_count": 1,
        "embedding_dimension": 384,
    }

    mocked_register.assert_called_once()


def test_process_document_no_usable_chunks():
    from pathlib import Path

    with (
        patch(
            "app.api.documents.save_uploaded_pdf",
            return_value=Path("data/uploads/empty.pdf"),
        ),
        patch(
            "app.api.documents.extract_text_by_page",
            return_value=[{"text": ""}],
        ),
        patch(
            "app.api.documents.save_extracted_text",
            return_value=Path("data/extracted/empty.json"),
        ),
        patch(
            "app.api.documents.create_chunks_from_extracted_json",
            return_value=[],
        ),
        patch(
            "app.api.documents.create_embeddings_for_chunks",
        ) as mocked_embeddings,
        patch(
            "app.api.documents.register_document",
        ) as mocked_register,
    ):
        response = TestClient(app).post(
            "/documents/process",
            files={
                "file": (
                    "empty.pdf",
                    b"%PDF-1.4 test content",
                    "application/pdf",
                )
            },
        )

    assert response.status_code == 400
    assert response.json() == {
        "detail": (
            "No usable text chunks were "
            "created from the document."
        )
    }

    mocked_embeddings.assert_not_called()
    mocked_register.assert_not_called()
