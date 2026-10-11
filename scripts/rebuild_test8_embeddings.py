
from pathlib import Path

import numpy as np

from app.services.document_registry import get_document
from app.services.embedding_service import (
    load_chunks,
    create_embeddings_for_chunks,
)


DOCUMENT_ID = "doc_260827cbbff7465c916b4f582fd69f05"
STAGING_DIR = Path("data_rebuild_test8")

document = get_document(DOCUMENT_ID)

if document is None:
    raise SystemExit("Document not found.")

staged_chunks_path = (
    STAGING_DIR
    / Path(document["chunks_file_path"]).name
)

if not staged_chunks_path.exists():
    raise SystemExit(
        f"Staged chunks not found: {staged_chunks_path}"
    )

chunks = load_chunks(staged_chunks_path)

print("Chunks loaded:", len(chunks))
print("Generating embeddings...")

embeddings = create_embeddings_for_chunks(chunks)

if embeddings.ndim != 2:
    raise RuntimeError("Embeddings must be a 2D array.")

if embeddings.shape[0] != len(chunks):
    raise RuntimeError(
        "Embedding count does not match chunk count."
    )

if not np.isfinite(embeddings).all():
    raise RuntimeError(
        "Embeddings contain NaN or infinite values."
    )

output_path = (
    STAGING_DIR
    / Path(document["embeddings_file_path"]).name
)

np.save(output_path, embeddings)

print("\nPASS: Embeddings generated successfully.")
print("Embedding shape:", embeddings.shape)
print("Embedding dtype:", embeddings.dtype)
print("Saved to:", output_path)
print("Original RAG files were NOT modified.")
