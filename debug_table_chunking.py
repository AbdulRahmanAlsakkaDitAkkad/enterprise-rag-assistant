
import fitz

from app.services.chunking_service import split_text_into_chunks
from app.services.pdf_service import extract_tables_from_page


PDF_PATH = (
    "data/uploads/"
    "b1442392-a79f-409a-aac4-2a7531827119_"
    "Grammatik aktiv Üben, Hören, Sprechen by Ute Voß.pdf"
)

with fitz.open(PDF_PATH) as document:
    tables = extract_tables_from_page(document[130])

    for index, table in enumerate(tables, start=1):
        if "können" not in table or "sollen" not in table:
            continue

        print(f"\n========== TABLE {index} ==========")

        chunks = split_text_into_chunks(table)

        print("Number of chunks:", len(chunks))

        for chunk_index, chunk in enumerate(chunks, start=1):
            print(f"\n--- CHUNK {chunk_index} ---")
            print(chunk)

print("\nExisting RAG files were NOT modified.")
