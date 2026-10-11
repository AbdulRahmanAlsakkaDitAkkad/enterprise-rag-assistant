
from pathlib import Path

import faiss
import numpy as np

from app.services.document_registry import get_document
from app.services.faiss_service import create_faiss_index


DOCUMENT_ID = "doc_260827cbbff7465c916b4f582fd69f05"
STAGING_DIR = Path("data_rebuild_test8")

document = get_document(DOCUMENT_ID)

if document is None:
    raise SystemExit("Document not found.")

embeddings_path = (
    STAGING_DIR
    / Path(document["embeddings_file_path"]).name
)

if not embeddings_path.exists():
    raise SystemExit(
        f"Staged embeddings not found: {embeddings_path}"
    )

embeddings = np.load(
    embeddings_path,
    allow_pickle=False,
)

print("Embedding shape:", embeddings.shape)

index = create_faiss_index(embeddings)

if index.ntotal != embeddings.shape[0]:
    raise RuntimeError(
        "FAISS vector count does not match embeddings."
    )

if index.d != embeddings.shape[1]:
    raise RuntimeError(
        "FAISS dimension does not match embeddings."
    )

# Save only to the staging directory.
output_path = (
    STAGING_DIR
    / Path(document["faiss_index_path"]).name
)

faiss.write_index(index, str(output_path))

# Reload the index to verify the saved file.
reloaded_index = faiss.read_index(str(output_path))

if reloaded_index.ntotal != len(embeddings):
    raise RuntimeError(
        "Reloaded FAISS index has an incorrect vector count."
    )

if reloaded_index.d != embeddings.shape[1]:
    raise RuntimeError(
        "Reloaded FAISS index has an incorrect dimension."
    )

print("\nPASS: Staged FAISS index created and validated.")
print("FAISS vectors:", reloaded_index.ntotal)
print("FAISS dimensions:", reloaded_index.d)
print("FAISS index type:", type(reloaded_index).__name__)
print("Saved to:", output_path)
print("Original RAG files were NOT modified.")
