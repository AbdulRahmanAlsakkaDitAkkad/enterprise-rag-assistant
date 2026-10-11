
from pathlib import Path

from app.services.document_registry import get_document
from app.services.pdf_service import extract_text_by_page

DOCUMENT_ID = "doc_260827cbbff7465c916b4f582fd69f05"

document = get_document(DOCUMENT_ID)

if document is None:
    raise SystemExit("Document not found.")

extracted_path = Path(document["extracted_file_path"])
pdf_path = Path("data/uploads") / (
    extracted_path.stem + ".pdf"
)

if not pdf_path.exists():
    raise SystemExit(f"Original PDF not found: {pdf_path}")

print("Re-extracting PDF with updated OCR settings...")

pages = extract_text_by_page(pdf_path)

page_131 = pages[130]

print("\nExtraction method:", page_131["extraction_method"])
print("\n========== PAGE 131 ==========\n")
print(page_131["text"])

print("\nExisting indexed files were NOT modified.")
