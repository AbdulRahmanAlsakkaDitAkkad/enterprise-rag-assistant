
import json
import re
import unicodedata
from pathlib import Path

from app.services.document_registry import get_document


DOCUMENT_ID = "doc_260827cbbff7465c916b4f582fd69f05"

question = (
    "Wie bildet man den Konjunktiv 2 "
    "von k\u00f6nnen und sollen?"
)


def tokenize(text):
    text = unicodedata.normalize("NFKC", text).casefold()
    return re.findall(r"[^\W_]+", text, flags=re.UNICODE)


STOPWORDS = {
    "wie", "bildet", "man", "den", "die", "das",
    "von", "und", "ein", "eine", "im", "in",
    "ist", "sind", "was", "werden", "wird",
    "der", "des", "dem", "zu", "mit", "auf",
}

query_terms = {
    term for term in tokenize(question)
    if term not in STOPWORDS and len(term) > 1
}

document = get_document(DOCUMENT_ID)

with Path(document["chunks_file_path"]).open(
    encoding="utf-8"
) as f:
    chunks = json.load(f)["chunks"]

results = []

for chunk in chunks:
    tokens = set(tokenize(chunk["text"]))

    matched = query_terms & tokens

    if not matched:
        continue

    coverage = len(matched) / len(query_terms)

    results.append(
        (coverage, len(matched), chunk, matched)
    )

results.sort(
    key=lambda item: (
        item[0],
        item[1],
    ),
    reverse=True,
)

print("Query terms:", sorted(query_terms))
print("Matching chunks:", len(results))
print("\nTop 15 keyword results:\n")

for rank, (coverage, count, chunk, matched) in enumerate(
    results[:15], start=1
):
    print(
        f"{rank}. "
        f"Coverage={coverage:.2f} "
        f"Page={chunk['page_number']} "
        f"Type={chunk.get('chunk_type', 'text')} "
        f"Chunk={chunk['chunk_index']} "
        f"Matched={sorted(matched)}"
    )

target = next(
    (
        rank
        for rank, (_, _, chunk, _) in enumerate(
            results, start=1
        )
        if chunk["page_number"] == 131
        and chunk["chunk_index"] == 6
    ),
    None,
)

print("\nTARGET TABLE KEYWORD RANK:", target)
print("Production RAG files were NOT modified.")
