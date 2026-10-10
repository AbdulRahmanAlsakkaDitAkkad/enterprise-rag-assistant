
import json
import uuid
from pathlib import Path

import cv2
import fitz
import numpy as np
import pytesseract
from fastapi import UploadFile
from PIL import Image


UPLOAD_DIR = Path("data/uploads")
EXTRACTED_DIR = Path("data/extracted")

OCR_LANGUAGES = "deu+eng"
OCR_SCALE = 3

MIN_TABLE_WIDTH = 250
MIN_TABLE_HEIGHT = 100


def save_uploaded_pdf(file: UploadFile) -> Path:
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise ValueError("Only PDF files are allowed.")

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    unique_name = f"{uuid.uuid4()}_{file.filename}"
    file_path = UPLOAD_DIR / unique_name

    with open(file_path, "wb") as buffer:
        buffer.write(file.file.read())

    return file_path


def render_page(page) -> Image.Image:
    pix = page.get_pixmap(
        matrix=fitz.Matrix(OCR_SCALE, OCR_SCALE),
        colorspace=fitz.csRGB,
        alpha=False,
    )

    return Image.frombytes(
        "RGB",
        (pix.width, pix.height),
        pix.samples,
    )


def extract_text_directly(page) -> str:
    return page.get_text("text", sort=True).strip()


def extract_text_with_ocr(page) -> str:
    image = render_page(page)

    return pytesseract.image_to_string(
        image,
        lang=OCR_LANGUAGES,
        config="--psm 3",
    ).strip()


def group_positions(positions, max_gap=12):
    groups = []

    for position in positions:
        position = int(position)

        if not groups or position - groups[-1][-1] > max_gap:
            groups.append([position])
        else:
            groups[-1].append(position)

    return [
        round(sum(group) / len(group))
        for group in groups
    ]


def detect_table_lines(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    binary = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        15,
        10,
    )

    horizontal_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (80, 1),
    )

    vertical_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (1, 80),
    )

    horizontal = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        horizontal_kernel,
    )

    vertical = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        vertical_kernel,
    )

    return horizontal, vertical


def detect_table_regions(image):
    horizontal, vertical = detect_table_lines(image)

    table_mask = cv2.add(horizontal, vertical)

    contours, _ = cv2.findContours(
        table_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    regions = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)

        if w < MIN_TABLE_WIDTH or h < MIN_TABLE_HEIGHT:
            continue

        regions.append((x, y, w, h))

    regions.sort(key=lambda region: (region[1], region[0]))

    return regions


def extract_table_cells(table_image):
    horizontal, vertical = detect_table_lines(table_image)

    horizontal_projection = horizontal.sum(axis=1)
    vertical_projection = vertical.sum(axis=0)

    row_positions = [
        index
        for index, value in enumerate(horizontal_projection)
        if value > 100000
    ]

    column_positions = [
        index
        for index, value in enumerate(vertical_projection)
        if value > 100000
    ]

    row_lines = group_positions(row_positions)
    column_lines = group_positions(column_positions)

    height, width = table_image.shape[:2]

    row_lines = sorted(set([0, *row_lines, height]))
    column_lines = sorted(set([0, *column_lines, width]))

    if len(row_lines) < 3 or len(column_lines) < 3:
        return []

    rows = []

    for row_index in range(len(row_lines) - 1):
        cells = []

        for col_index in range(len(column_lines) - 1):
            x1 = column_lines[col_index] + 8
            x2 = column_lines[col_index + 1] - 8
            y1 = row_lines[row_index] + 8
            y2 = row_lines[row_index + 1] - 8

            if x2 <= x1 or y2 <= y1:
                cells.append("")
                continue

            cell_image = table_image[y1:y2, x1:x2]

            rgb = cv2.cvtColor(
                cell_image,
                cv2.COLOR_BGR2RGB,
            )

            text = pytesseract.image_to_string(
                Image.fromarray(rgb),
                lang=OCR_LANGUAGES,
                config="--psm 6",
            ).strip()

            cells.append(" ".join(text.split()))

        rows.append(cells)

    return rows


def extract_tables_from_page(page):
    pil_image = render_page(page)

    image = cv2.cvtColor(
        np.array(pil_image),
        cv2.COLOR_RGB2BGR,
    )

    regions = detect_table_regions(image)

    extracted_tables = []

    for x, y, w, h in regions:
        table_image = image[y:y + h, x:x + w]

        rows = extract_table_cells(table_image)

        if len(rows) < 2:
            continue

        if not any(
            any(cell.strip() for cell in row)
            for row in rows
        ):
            continue

        table_text = "\n".join(
            " | ".join(row)
            for row in rows
        )

        extracted_tables.append(table_text)

    return extracted_tables


def extract_text_by_page(pdf_path: Path) -> list[dict]:
    pages = []

    with fitz.open(pdf_path) as document:
        for page_index in range(len(document)):
            page = document[page_index]

            direct_text = extract_text_directly(page)

            if direct_text:
                final_text = direct_text
                extraction_method = "direct"
            else:
                final_text = extract_text_with_ocr(page)
                extraction_method = "ocr"

            tables = extract_tables_from_page(page)

            if tables:
                table_section = "\n\n".join(
                    f"[TABLE {index}]\n{table}"
                    for index, table in enumerate(tables, start=1)
                )

                final_text = (
                    final_text
                    + "\n\nSTRUCTURED TABLES:\n"
                    + table_section
                )

                extraction_method += "+tables"

            pages.append(
                {
                    "page_number": page_index + 1,
                    "text": final_text.strip(),
                    "extraction_method": extraction_method,
                }
            )

    return pages


def save_extracted_text(pdf_path: Path, pages: list[dict]) -> Path:
    EXTRACTED_DIR.mkdir(parents=True, exist_ok=True)

    output_path = EXTRACTED_DIR / f"{pdf_path.stem}.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "source_file": pdf_path.name,
                "pages": pages,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    return output_path
