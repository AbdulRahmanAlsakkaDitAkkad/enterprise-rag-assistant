
from pathlib import Path

import fitz
import pytesseract
from PIL import Image


# 1. Find uploaded PDF files
pdf_files = sorted(Path("data/uploads").glob("*.pdf"))

if not pdf_files:
    raise SystemExit("No PDF files found in data/uploads.")

print("\nAvailable PDFs:")

for index, path in enumerate(pdf_files, start=1):
    print(f"{index}. {path.name}")


# 2. Select PDF
try:
    choice = int(input("\nSelect PDF number: ")) - 1
except ValueError:
    raise SystemExit("Please enter a valid number.")

if not 0 <= choice < len(pdf_files):
    raise SystemExit("Invalid PDF selection.")

pdf_path = pdf_files[choice]


# 3. Open page 131
with fitz.open(pdf_path) as document:

    page_index = 130

    if page_index >= len(document):
        raise SystemExit("PDF does not contain page 131.")

    page = document[page_index]

    print("\n========== PAGE INFORMATION ==========")
    print("PDF:", pdf_path.name)
    print("Page:", page_index + 1)
    print("Page size:", page.rect)


    # 4. Render page and save image
    pix = page.get_pixmap(
        matrix=fitz.Matrix(3, 3),
        colorspace=fitz.csRGB,
        alpha=False,
    )

    output_path = Path("debug_page_131.png")
    pix.save(str(output_path))

    print("\n========== IMAGE SAVED ==========")
    print("File:", output_path.resolve())


    # 5. Convert rendered page to PIL image
    image = Image.frombytes(
        "RGB",
        (pix.width, pix.height),
        pix.samples,
    )


    # 6. Run German OCR using PSM 3
    print("\n========== GERMAN OCR (PSM 3) ==========\n")

    text = pytesseract.image_to_string(
        image,
        lang="deu",
        config="--psm 3",
    )

    print(text)


    # 7. Finish
    print("\n========== FINISHED ==========")
    print("Page image:", output_path.resolve())
    print("No existing RAG files were modified.")
