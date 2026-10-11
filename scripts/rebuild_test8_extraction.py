
import json
import time
from pathlib import Path

from app.services.document_registry import get_document
from app.services.pdf_service import extract_text_by_page


DOCUMENT_ID = "doc_260827cbbff7465c916b4f582fd69f05"
STAGING_DIR = Path("data_rebuild_test8")

document = get_document(DOCUMENT_ID)

if document is None:
    raise SystemExit("Document not found.")

original_extracted_path = Path(document["extracted_file_path"])
pdf_path = Path("data/uploads") / (
    original_extracted_path.stem + ".pdf"
)

if not pdf_path.exists():
    raise SystemExit(f"PDF not found: {pdf_path}")

STAGING_DIR.mkdir(parents=True, exist_ok=True)

output_path = STAGING_DIR / original_extracted_path.name

print("PDF:", pdf_path)
print("Staging output:", output_path)
print("Starting extraction...")

start = time.time()

pages = extract_text_by_page(pdf_path)

if len(pages) != 258:
    raise RuntimeError(
        f"Expected 258 pages, got {len(pages)}"
    )

data = {
    "source_file": pdf_path.name,
    "pages": pages,
}

with output_path.open("w", encoding="utf-8") as f:
    json.dump(
        data,
        f,
        ensure_ascii=False,
        indent=2,
    )

page_131 = next(
    page for page in pages
    if page["page_number"] == 131
)

text = page_131["text"]

print("\nExtraction completed.")
print("Pages:", len(pages))
print("Elapsed seconds:", round(time.time() - start, 1))
print("Page 131 has structured tables:",
      "STRUCTURED TABLES:" in text)
print("Page 131 contains 'könntest':",
      "könntest" in text)
print("Page 131 contains 'solltest':",
      "solltest" in text)
print("Saved to:", output_path)
print("Original RAG files were NOT modified.")
