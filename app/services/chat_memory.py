import sqlite3
import uuid
from pathlib import Path


DATABASE_PATH = Path("data/documents.db")


def get_connection() -> sqlite3.Connection:
    """
    Reuse the same SQLite database file used by the document registry.
    """
    DATABASE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    connection = sqlite3.connect(
        DATABASE_PATH
    )

    connection.row_factory = sqlite3.Row

    return connection


def initialize_chat_tables() -> None:
    """
    Create chat session and message tables if they do not exist.
    """
    with get_connection() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS chat_sessions (
                session_id TEXT PRIMARY KEY,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS chat_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (session_id)
                    REFERENCES chat_sessions(session_id)
            )
            """
        )

        connection.commit()


def create_session() -> str:
    """
    Create and return a new chat session ID.
    """
    initialize_chat_tables()

    session_id = f"session_{uuid.uuid4().hex}"

    with get_connection() as connection:
        connection.execute(
            """
            INSERT INTO chat_sessions (
                session_id
            )
            VALUES (?)
            """,
            (
                session_id,
            ),
        )

        connection.commit()

    return session_id


def session_exists(
    session_id: str,
) -> bool:
    """
    Check whether a chat session exists.
    """
    initialize_chat_tables()

    with get_connection() as connection:
        row = connection.execute(
            """
            SELECT session_id
            FROM chat_sessions
            WHERE session_id = ?
            """,
            (
                session_id,
            ),
        ).fetchone()

    return row is not None


def add_message(
    session_id: str,
    role: str,
    content: str,
) -> None:
    """
    Save one chat message.

    role should normally be:
    - user
    - assistant
    """
    initialize_chat_tables()

    if not session_exists(session_id):
        raise ValueError(
            f"Chat session does not exist: {session_id}"
        )

    if role not in {
        "user",
        "assistant",
    }:
        raise ValueError(
            "Role must be 'user' or 'assistant'."
        )

    cleaned_content = content.strip()

    if not cleaned_content:
        raise ValueError(
            "Message content cannot be empty."
        )

    with get_connection() as connection:
        connection.execute(
            """
            INSERT INTO chat_messages (
                session_id,
                role,
                content
            )
            VALUES (?, ?, ?)
            """,
            (
                session_id,
                role,
                cleaned_content,
            ),
        )

        connection.commit()


def get_recent_messages(
    session_id: str,
    limit: int = 6,
) -> list[dict]:
    """
    Return recent messages in chronological order.
    """
    initialize_chat_tables()

    if not session_exists(session_id):
        raise ValueError(
            f"Chat session does not exist: {session_id}"
        )

    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT
                role,
                content,
                created_at
            FROM chat_messages
            WHERE session_id = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (
                session_id,
                limit,
            ),
        ).fetchall()

    messages = [
        dict(row)
        for row in reversed(rows)
    ]

    return messages


def clear_session(
    session_id: str,
) -> None:
    """
    Delete all messages for a session,
    but keep the session itself.
    """
    initialize_chat_tables()

    if not session_exists(session_id):
        raise ValueError(
            f"Chat session does not exist: {session_id}"
        )

    with get_connection() as connection:
        connection.execute(
            """
            DELETE FROM chat_messages
            WHERE session_id = ?
            """,
            (
                session_id,
            ),
        )

        connection.commit()