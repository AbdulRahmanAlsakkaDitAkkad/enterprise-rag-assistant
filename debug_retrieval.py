from pathlib import Path

from app.services.document_registry import get_document
from app.services.faiss_service import search_faiss_index


DOCUMENT_ID = (
    "doc_260827cbbff7465c916b4f582fd69f05"
)

QUERY = (
    "sollen Bedeutung Gebrauch "
    "Erklärung Regel Grammatik"
)


document = get_document(
    DOCUMENT_ID
)

if document is None:
    raise RuntimeError(
        "Document not found."
    )


index_path = Path(
    document["faiss_index_path"]
)

chunks_path = Path(
    document["chunks_file_path"]
)


results = search_faiss_index(
    query=QUERY,
    index_path=index_path,
    chunks_file_path=chunks_path,
    top_k=10,
)


print()
print(
    "========== FINAL SEARCH RESULTS =========="
)


for rank, result in enumerate(
    results,
    start=1,
):
    print()
    print(
        "----------------------------------------"
    )

    print(
        f"RANK: {rank}"
    )

    print(
        f"PAGE: {result['page_number']}"
    )

    print(
        f"CHUNK INDEX: {result['chunk_index']}"
    )

    print(
        "RETRIEVAL TYPE: "
        f"{result.get('retrieval_type', 'unknown')}"
    )

    print(
        f"FAISS: {result['score']:.4f}"
    )

    print(
        "RERANKER: "
        f"{result['reranker_score']:.4f}"
    )

    text = result[
        "text"
    ].replace(
        "\n",
        " ",
    )

    print(
        "TEXT:",
        text[:500],
    )


print()
print(
    "=========================================="
)