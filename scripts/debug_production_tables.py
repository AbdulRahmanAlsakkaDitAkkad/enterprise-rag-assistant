
from pathlib import Path

import fitz

from app.services.pdf_service import extract_tables_from_page


PDF_PATH = Path(
    "data/uploads/"
    "b1442392-a79f-409a-aac4-2a7531827119_"
    "Grammatik aktiv Üben, Hören, Sprechen by Ute Voß.pdf"
)

if not PDF_PATH.exists():
    raise SystemExit(f"PDF not found: {PDF_PATH}")

with fitz.open(PDF_PATH) as document:
    page = document[130]

    print("Extracting tables from page 131...")

    tables = extract_tables_from_page(page)

    print(f"\nDetected tables: {len(tables)}")

    for index, table in enumerate(tables, start=1):
        print(f"\n========== TABLE {index} ==========\n")
        print(table)

print("\nFinished. Existing RAG files were NOT modified.")
