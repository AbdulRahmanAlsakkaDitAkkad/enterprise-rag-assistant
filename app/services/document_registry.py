import sqlite3
import uuid
from pathlib import Path


DATABASE_PATH = Path("data/documents.db")


def get_connection() -> sqlite3.Connection:
    DATABASE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    connection = sqlite3.connect(
        DATABASE_PATH
    )

    connection.row_factory = sqlite3.Row

    return connection


def initialize_database() -> None:
    with get_connection() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (
                document_id TEXT PRIMARY KEY,
                original_file_name TEXT NOT NULL,
                extracted_file_path TEXT NOT NULL,
                chunks_file_path TEXT NOT NULL,
                embeddings_file_path TEXT NOT NULL,
                faiss_index_path TEXT NOT NULL
            )
            """
        )

        connection.commit()


def generate_document_id() -> str:
    return f"doc_{uuid.uuid4().hex}"


def register_document(
    original_file_name: str,
    extracted_file_path: str,
    chunks_file_path: str,
    embeddings_file_path: str,
    faiss_index_path: str,
) -> str:
    initialize_database()

    document_id = generate_document_id()

    with get_connection() as connection:
        connection.execute(
            """
            INSERT INTO documents (
                document_id,
                original_file_name,
                extracted_file_path,
                chunks_file_path,
                embeddings_file_path,
                faiss_index_path
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                document_id,
                original_file_name,
                extracted_file_path,
                chunks_file_path,
                embeddings_file_path,
                faiss_index_path,
            ),
        )

        connection.commit()

    return document_id


def get_document(
    document_id: str,
) -> dict | None:
    initialize_database()

    with get_connection() as connection:
        row = connection.execute(
            """
            SELECT
                document_id,
                original_file_name,
                extracted_file_path,
                chunks_file_path,
                embeddings_file_path,
                faiss_index_path
            FROM documents
            WHERE document_id = ?
            """,
            (
                document_id,
            ),
        ).fetchone()

    if row is None:
        return None

    return dict(row)