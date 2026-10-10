
from pathlib import Path

import fitz

from app.services.pdf_service import (
    extract_text_directly,
    extract_text_with_ocr,
    extract_tables_from_page,
)
from app.services.chunking_service import split_page_into_chunks


PDF_PATH = Path(
    "data/uploads/"
    "b1442392-a79f-409a-aac4-2a7531827119_"
    "Grammatik aktiv Üben, Hören, Sprechen by Ute Voß.pdf"
)

if not PDF_PATH.exists():
    raise SystemExit(f"PDF not found: {PDF_PATH}")

with fitz.open(PDF_PATH) as document:
    page = document[130]

    text = extract_text_directly(page)

    if not text:
        text = extract_text_with_ocr(page)

    tables = extract_tables_from_page(page)

    if tables:
        table_section = "\n\n".join(
            f"[TABLE {index}]\n{table}"
            for index, table in enumerate(tables, start=1)
        )
        text += "\n\nSTRUCTURED TABLES:\n" + table_section

chunks = split_page_into_chunks(text)

print("\nTotal chunks:", len(chunks))

for index, chunk in enumerate(chunks, start=1):
    print(
        f"Chunk {index}: "
        f"type={chunk['chunk_type']}, "
        f"length={len(chunk['text'])}"
    )

    if chunk["chunk_type"] == "table" and (
        "können" in chunk["text"]
        and "sollen" in chunk["text"]
    ):
        print("\n========== MODAL VERBS TABLE ==========\n")
        print(chunk["text"])

        required = [
            "könnte",
            "könntest",
            "könnten",
            "könntet",
            "sollte",
            "solltest",
            "sollten",
            "solltet",
        ]

        missing = [
            word for word in required
            if word not in chunk["text"]
        ]

        print("\nMissing required forms:", missing)

print("\nExisting RAG files were NOT modified.")
