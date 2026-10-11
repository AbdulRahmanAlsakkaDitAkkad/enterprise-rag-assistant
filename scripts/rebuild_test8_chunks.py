
import json
from pathlib import Path

from app.services.document_registry import get_document
from app.services.chunking_service import (
    create_chunks_from_extracted_json,
)

DOCUMENT_ID = "doc_260827cbbff7465c916b4f582fd69f05"
STAGING_DIR = Path("data_rebuild_test8")

document = get_document(DOCUMENT_ID)

if document is None:
    raise SystemExit("Document not found.")

original_extracted_path = Path(
    document["extracted_file_path"]
)

staged_extracted_path = (
    STAGING_DIR / original_extracted_path.name
)

if not staged_extracted_path.exists():
    raise SystemExit(
        f"Staged extraction not found: {staged_extracted_path}"
    )

chunks = create_chunks_from_extracted_json(
    staged_extracted_path
)

if not chunks:
    raise RuntimeError("No chunks were generated.")

output_path = (
    STAGING_DIR
    / Path(document["chunks_file_path"]).name
)

with output_path.open("w", encoding="utf-8") as f:
    json.dump(
        {
            "source_file": staged_extracted_path.name,
            "chunk_count": len(chunks),
            "chunks": chunks,
        },
        f,
        ensure_ascii=False,
        indent=2,
    )

page_131_chunks = [
    chunk for chunk in chunks
    if chunk["page_number"] == 131
]

modal_tables = [
    chunk for chunk in page_131_chunks
    if chunk.get("chunk_type") == "table"
    and "können" in chunk["text"]
    and "sollen" in chunk["text"]
]

required_forms = [
    "könnte",
    "könntest",
    "könnten",
    "könntet",
    "sollte",
    "solltest",
    "sollten",
    "solltet",
]

print("Total chunks:", len(chunks))
print("Page 131 chunks:", len(page_131_chunks))
print("Modal verb tables found:", len(modal_tables))

if len(modal_tables) != 1:
    raise RuntimeError(
        "Expected exactly one modal verb table on page 131."
    )

modal_table = modal_tables[0]

missing = [
    form for form in required_forms
    if form not in modal_table["text"]
]

print("Modal table chunk ID:", modal_table["chunk_id"])
print("Modal table length:", len(modal_table["text"]))
print("Missing required forms:", missing)

if missing:
    raise RuntimeError(
        f"Modal verb table is incomplete: {missing}"
    )

print("\nPASS: Complete modal verb table preserved.")
print("Staged chunks saved to:", output_path)
print("Original RAG files were NOT modified.")
