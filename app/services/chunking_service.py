
import json
import re
from pathlib import Path


CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

TABLE_SECTION_MARKER = "STRUCTURED TABLES:"

TABLE_MARKER_PATTERN = re.compile(
    r"(?m)^\[TABLE\s+\d+\]\s*$"
)


def split_text_into_chunks(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """
    Split ordinary text into overlapping character chunks.

    Structured tables are handled separately by
    split_page_into_chunks().
    """
    if not text or not text.strip():
        return []

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError(
            "overlap must be non-negative and smaller than chunk_size."
        )

    chunks = []
    start = 0
    step = chunk_size - overlap

    while start < len(text):
        end = min(start + chunk_size, len(text))

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        start += step

    return chunks


def split_page_into_chunks(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[dict]:
    """
    Split a page into ordinary text chunks and complete table chunks.

    Expected extraction format:

        Ordinary page text...

        STRUCTURED TABLES:
        [TABLE 1]
        header | header
        value | value

        [TABLE 2]
        header | header
        value | value

    Each table is kept as one atomic chunk.
    """
    if not text or not text.strip():
        return []

    chunks = []

    # Separate ordinary page text from structured tables.
    if TABLE_SECTION_MARKER not in text:
        return [
            {
                "text": chunk,
                "chunk_type": "text",
            }
            for chunk in split_text_into_chunks(
                text,
                chunk_size,
                overlap,
            )
        ]

    ordinary_text, table_section = text.split(
        TABLE_SECTION_MARKER,
        1,
    )

    # 1. Chunk ordinary text normally.
    for chunk in split_text_into_chunks(
        ordinary_text,
        chunk_size,
        overlap,
    ):
        chunks.append(
            {
                "text": chunk,
                "chunk_type": "text",
            }
        )

    # 2. Find structured table markers.
    matches = list(
        TABLE_MARKER_PATTERN.finditer(table_section)
    )

    if not matches:
        # Avoid silently losing content if table markers
        # are missing or malformed.
        for chunk in split_text_into_chunks(
            table_section,
            chunk_size,
            overlap,
        ):
            chunks.append(
                {
                    "text": chunk,
                    "chunk_type": "text",
                }
            )

        return chunks

    # Preserve any text before the first table marker.
    prefix = table_section[:matches[0].start()].strip()

    if prefix:
        for chunk in split_text_into_chunks(
            prefix,
            chunk_size,
            overlap,
        ):
            chunks.append(
                {
                    "text": chunk,
                    "chunk_type": "text",
                }
            )

    # 3. Keep each table as one complete chunk.
    for index, match in enumerate(matches):
        start = match.start()

        if index + 1 < len(matches):
            end = matches[index + 1].start()
        else:
            end = len(table_section)

        table_text = table_section[start:end].strip()

        if not table_text:
            continue

        chunks.append(
            {
                "text": table_text,
                "chunk_type": "table",
            }
        )

    return chunks


def create_chunks_from_extracted_json(
    extracted_json_path: Path,
) -> list[dict]:
    with open(
        extracted_json_path,
        "r",
        encoding="utf-8",
    ) as f:
        data = json.load(f)

    source_file = data["source_file"]
    all_chunks = []

    for page in data["pages"]:
        page_number = page["page_number"]
        text = page.get("text", "")
        extraction_method = page.get(
            "extraction_method",
            "unknown",
        )

        page_chunks = split_page_into_chunks(text)

        for chunk_index, chunk_data in enumerate(
            page_chunks,
            start=1,
        ):
            chunk_id = (
                f"{Path(source_file).stem}"
                f"_p{page_number}"
                f"_c{chunk_index}"
            )

            all_chunks.append(
                {
                    "chunk_id": chunk_id,
                    "source_file": source_file,
                    "page_number": page_number,
                    "chunk_index": chunk_index,
                    "text": chunk_data["text"],
                    "chunk_type": chunk_data["chunk_type"],
                    "extraction_method": extraction_method,
                }
            )

    return all_chunks


def save_chunks(
    extracted_json_path: Path,
    chunks: list[dict],
) -> Path:
    output_dir = Path("data/extracted")
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        output_dir
        / f"{extracted_json_path.stem}_chunks.json"
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            {
                "source_file": extracted_json_path.name,
                "chunk_count": len(chunks),
                "chunks": chunks,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    return output_path
