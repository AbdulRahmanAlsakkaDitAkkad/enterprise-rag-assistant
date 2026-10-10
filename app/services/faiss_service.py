
import hashlib
import json
import re
import unicodedata
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import (
    CrossEncoder,
    SentenceTransformer,
)


INDEX_DIR = Path("data/indexes")

EMBEDDING_MODEL_NAME = (
    "sentence-transformers/"
    "paraphrase-multilingual-MiniLM-L12-v2"
)

RERANKER_MODEL_NAME = (
    "cross-encoder/"
    "mmarco-mMiniLMv2-L12-H384-v1"
)

MIN_CHUNK_LENGTH = 100
MIN_FAISS_SCORE = 0.30

DEFAULT_CANDIDATE_COUNT = 100
DEFAULT_KEYWORD_CANDIDATE_COUNT = 50
DEFAULT_NEIGHBOR_DISTANCE = 1


embedding_model = SentenceTransformer(
    EMBEDDING_MODEL_NAME
)

reranker_model = CrossEncoder(
    RERANKER_MODEL_NAME
)


def create_faiss_index(
    embeddings: np.ndarray,
):
    """
    Create a FAISS inner-product index.

    Embeddings are normalized, so inner product
    behaves like cosine similarity.
    """
    if embeddings.ndim != 2:
        raise ValueError(
            "Embeddings must be a two-dimensional array."
        )

    if embeddings.shape[0] == 0:
        raise ValueError(
            "Cannot create a FAISS index without embeddings."
        )

    embeddings = embeddings.astype("float32")

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    return index


def create_safe_index_name(
    chunks_file_path: Path,
) -> str:
    """
    Create a deterministic ASCII-only filename.

    This avoids Unicode filename issues with FAISS
    on Windows.
    """
    path_string = str(chunks_file_path)

    path_hash = hashlib.sha256(
        path_string.encode("utf-8")
    ).hexdigest()[:16]

    return f"faiss_{path_hash}.faiss"


def save_faiss_index(
    index,
    chunks_file_path: Path,
) -> Path:
    """
    Save a FAISS index using an ASCII-safe filename.
    """
    INDEX_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    safe_file_name = create_safe_index_name(
        chunks_file_path
    )

    output_path = INDEX_DIR / safe_file_name

    faiss.write_index(
        index,
        str(output_path),
    )

    return output_path


def load_embeddings(
    embeddings_path: Path,
) -> np.ndarray:
    """
    Load embeddings from a NumPy file.
    """
    if not embeddings_path.exists():
        raise FileNotFoundError(
            "Embeddings file not found: "
            f"{embeddings_path}"
        )

    embeddings = np.load(embeddings_path)

    return embeddings.astype("float32")


def load_chunks(
    chunks_file_path: Path,
) -> list[dict]:
    """
    Load chunks and metadata from JSON.
    """
    if not chunks_file_path.exists():
        raise FileNotFoundError(
            "Chunks file not found: "
            f"{chunks_file_path}"
        )

    with open(
        chunks_file_path,
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    chunks = data.get("chunks", [])

    if not chunks:
        raise ValueError(
            "The chunks file contains no chunks."
        )

    return chunks


def load_faiss_index(
    index_path: Path,
):
    """
    Load a saved FAISS index.
    """
    if not index_path.exists():
        raise FileNotFoundError(
            "FAISS index file not found: "
            f"{index_path}"
        )

    return faiss.read_index(
        str(index_path)
    )


def build_candidate(
    chunk: dict,
    score: float,
    retrieval_type: str = "semantic",
) -> dict:
    """
    Convert a stored chunk into a retrieval candidate.
    """
    chunk_text = chunk.get(
        "text",
        "",
    ).strip()

    return {
        "score": float(score),
        "chunk_id": chunk["chunk_id"],
        "source_file": chunk["source_file"],
        "page_number": chunk["page_number"],
        "chunk_index": chunk["chunk_index"],
        "text": chunk_text,
        "extraction_method": chunk.get(
            "extraction_method",
            "unknown",
        ),
        "retrieval_type": retrieval_type,
    }


def retrieve_faiss_candidates(
    query: str,
    index_path: Path,
    chunks_file_path: Path,
    candidate_count: int = DEFAULT_CANDIDATE_COUNT,
) -> list[dict]:
    """
    Retrieve candidate chunks using FAISS only.

    No CrossEncoder reranking happens here.
    """
    cleaned_query = query.strip()

    if not cleaned_query:
        raise ValueError(
            "The search query cannot be empty."
        )

    index = load_faiss_index(index_path)
    chunks = load_chunks(chunks_file_path)

    if index.ntotal != len(chunks):
        raise ValueError(
            "The FAISS index and chunks file do not "
            "contain the same number of items."
        )

    query_embedding = embedding_model.encode(
        [cleaned_query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    candidate_count = min(
        max(candidate_count, 0),
        index.ntotal,
    )

    if candidate_count == 0:
        return []

    scores, indices = index.search(
        query_embedding,
        candidate_count,
    )

    candidates = []

    for score, idx in zip(
        scores[0],
        indices[0],
    ):
        if idx == -1:
            continue

        faiss_score = float(score)

        if faiss_score < MIN_FAISS_SCORE:
            continue

        chunk = chunks[int(idx)]

        chunk_text = chunk.get(
            "text",
            "",
        ).strip()

        if len(chunk_text) < MIN_CHUNK_LENGTH:
            continue

        candidates.append(
            build_candidate(
                chunk=chunk,
                score=faiss_score,
                retrieval_type="semantic",
            )
        )

    return candidates


# --------------------------------------------------
# Keyword retrieval
# --------------------------------------------------

KEYWORD_STOPWORDS = {
    "wie",
    "bildet",
    "man",
    "den",
    "die",
    "das",
    "von",
    "und",
    "ein",
    "eine",
    "im",
    "in",
    "ist",
    "sind",
    "was",
    "werden",
    "wird",
    "der",
    "des",
    "dem",
    "zu",
    "mit",
    "auf",
}


def tokenize_keywords(
    text: str,
) -> set[str]:
    """
    Extract case-insensitive Unicode word tokens.

    German umlauts are preserved.
    """
    normalized = unicodedata.normalize(
        "NFKC",
        text,
    ).casefold()

    return set(
        re.findall(
            r"[^\W_]+",
            normalized,
            flags=re.UNICODE,
        )
    )


def retrieve_keyword_candidates(
    query: str,
    chunks_file_path: Path,
    candidate_count: int = (
        DEFAULT_KEYWORD_CANDIDATE_COUNT
    ),
) -> list[dict]:
    """
    Retrieve candidates using exact keyword overlap.

    Keyword retrieval complements semantic retrieval
    and does not apply MIN_FAISS_SCORE.

    The CrossEncoder performs final ranking.
    """
    cleaned_query = query.strip()

    if not cleaned_query:
        raise ValueError(
            "The search query cannot be empty."
        )

    if candidate_count < 1:
        return []

    query_terms = {
        term
        for term in tokenize_keywords(cleaned_query)
        if (
            term not in KEYWORD_STOPWORDS
            and len(term) > 1
        )
    }

    if not query_terms:
        return []

    chunks = load_chunks(chunks_file_path)

    matches = []

    for chunk in chunks:
        chunk_text = chunk.get(
            "text",
            "",
        ).strip()

        if len(chunk_text) < MIN_CHUNK_LENGTH:
            continue

        chunk_terms = tokenize_keywords(
            chunk_text
        )

        matched_terms = (
            query_terms & chunk_terms
        )

        if not matched_terms:
            continue

        coverage = (
            len(matched_terms)
            / len(query_terms)
        )

        matches.append(
            (
                coverage,
                len(matched_terms),
                chunk,
            )
        )

    matches.sort(
        key=lambda item: (
            item[0],
            item[1],
        ),
        reverse=True,
    )

    return [
        build_candidate(
            chunk=chunk,
            score=coverage,
            retrieval_type="keyword",
        )
        for coverage, _, chunk in matches[
            :candidate_count
        ]
    ]


# --------------------------------------------------
# Hybrid candidate merging
# --------------------------------------------------

def merge_candidates(
    semantic_candidates: list[dict],
    keyword_candidates: list[dict],
) -> list[dict]:
    """
    Merge semantic and keyword candidates.

    Deduplicate using chunk_id.

    Keep original retrieval scores for diagnostics;
    scores from different retrieval methods are
    not directly compared.
    """
    merged = {}

    for candidate in semantic_candidates:
        chunk_id = candidate["chunk_id"]

        if chunk_id not in merged:
            merged[chunk_id] = candidate.copy()

    for candidate in keyword_candidates:
        chunk_id = candidate["chunk_id"]

        if chunk_id in merged:
            merged[chunk_id]["retrieval_type"] = (
                "hybrid"
            )
        else:
            merged[chunk_id] = candidate.copy()

    return list(merged.values())


# --------------------------------------------------
# Neighbor expansion
# --------------------------------------------------

def add_neighbor_chunks(
    candidates: list[dict],
    chunks_file_path: Path,
    neighbor_distance: int = (
        DEFAULT_NEIGHBOR_DISTANCE
    ),
) -> list[dict]:
    """
    Add neighboring chunks around retrieved candidates.

    chunk_index is local to a page, not globally unique.

    Neighbors are identified using:
        source_file + page_number + chunk_index
    """
    if not candidates:
        return []

    if neighbor_distance < 1:
        return candidates

    chunks = load_chunks(
        chunks_file_path
    )

    chunks_by_location = {}

    for chunk in chunks:
        key = (
            chunk["source_file"],
            chunk["page_number"],
            chunk["chunk_index"],
        )

        chunks_by_location[key] = chunk

    expanded = {}

    for candidate in candidates:
        expanded[
            candidate["chunk_id"]
        ] = candidate

    for candidate in candidates:
        current_file = candidate[
            "source_file"
        ]

        current_page = candidate[
            "page_number"
        ]

        current_index = candidate[
            "chunk_index"
        ]

        current_score = candidate.get(
            "score",
            0.0,
        )

        for offset in range(
            -neighbor_distance,
            neighbor_distance + 1,
        ):
            if offset == 0:
                continue

            neighbor_index = (
                current_index + offset
            )

            key = (
                current_file,
                current_page,
                neighbor_index,
            )

            neighbor_chunk = (
                chunks_by_location.get(key)
            )

            if neighbor_chunk is None:
                continue

            neighbor_text = (
                neighbor_chunk.get(
                    "text",
                    "",
                ).strip()
            )

            if (
                len(neighbor_text)
                < MIN_CHUNK_LENGTH
            ):
                continue

            neighbor_id = (
                neighbor_chunk["chunk_id"]
            )

            if neighbor_id in expanded:
                continue

            expanded[
                neighbor_id
            ] = build_candidate(
                chunk=neighbor_chunk,
                score=current_score,
                retrieval_type="neighbor",
            )

    return list(expanded.values())


# --------------------------------------------------
# CrossEncoder reranking
# --------------------------------------------------

def rerank_results(
    query: str,
    candidates: list[dict],
    top_k: int,
) -> list[dict]:
    """
    Globally rerank candidate chunks with the
    CrossEncoder.

    CrossEncoder scores are used for ordering only.
    """
    if not candidates:
        return []

    if top_k < 1:
        return []

    query_chunk_pairs = [
        (
            query,
            candidate["text"],
        )
        for candidate in candidates
    ]

    reranker_scores = reranker_model.predict(
        query_chunk_pairs,
        show_progress_bar=False,
    )

    for candidate, reranker_score in zip(
        candidates,
        reranker_scores,
    ):
        candidate[
            "reranker_score"
        ] = float(reranker_score)

    candidates.sort(
        key=lambda item: item.get(
            "reranker_score",
            float("-inf"),
        ),
        reverse=True,
    )

    return candidates[:top_k]



# --------------------------------------------------
# Query-relevant table evidence expansion
# --------------------------------------------------

def add_relevant_page_tables(
    query: str,
    ranked_results: list[dict],
    chunks_file_path: Path,
    max_tables: int = 2,
) -> list[dict]:
    """
    Add relevant tables from pages already represented
    in the ranked results.

    Existing results retain their ranking and order.
    Table evidence is added only when multiple meaningful
    query terms occur in the table text.
    """
    if not ranked_results or max_tables < 1:
        return ranked_results

    query_terms = tokenize_keywords(query) - {
        "formen", "form", "konjunktiv",
        "konjugation", "konjugiert",
    }

    if len(query_terms) < 2:
        return ranked_results

    pages = []
    for result in ranked_results:
        page = result["page_number"]
        if page not in pages:
            pages.append(page)

    chunks = load_chunks(chunks_file_path)
    existing_ids = {
        result["chunk_id"]
        for result in ranked_results
    }

    additions = []

    for page in pages:
        candidates = []

        for chunk in chunks:
            if chunk["page_number"] != page:
                continue

            if chunk.get("chunk_type") != "table":
                continue

            if chunk["chunk_id"] in existing_ids:
                continue

            table_terms = tokenize_keywords(chunk["text"])
            matched = query_terms & table_terms

            if len(matched) < 2:
                continue

            candidates.append((
                len(matched),
                len(chunk["text"]),
                chunk,
            ))

        candidates.sort(
            key=lambda item: (item[0], item[1]),
            reverse=True,
        )

        for _, _, chunk in candidates:
            candidate = build_candidate(
                chunk=chunk,
                score=0.0,
                retrieval_type="page_table",
            )
            candidate["reranker_score"] = None

            additions.append(candidate)
            existing_ids.add(chunk["chunk_id"])

            if len(additions) >= max_tables:
                return ranked_results + additions

    return ranked_results + additions


# --------------------------------------------------
# Hybrid search entry point
# --------------------------------------------------

def search_faiss_index(
    query: str,
    index_path: Path,
    chunks_file_path: Path,
    top_k: int = 5,
) -> list[dict]:
    """
    Search one document using hybrid retrieval.

    Flow:

        query
          -> FAISS semantic candidates
          -> keyword candidates
          -> merge and deduplicate
          -> neighboring chunk expansion
          -> CrossEncoder reranking
          -> top_k

    The public function signature is unchanged.
    """
    if top_k < 1:
        raise ValueError(
            "top_k must be at least 1."
        )

    if not query.strip():
        raise ValueError(
            "The search query cannot be empty."
        )

    candidate_count = max(
        top_k * 20,
        DEFAULT_CANDIDATE_COUNT,
    )

    semantic_candidates = (
        retrieve_faiss_candidates(
            query=query,
            index_path=index_path,
            chunks_file_path=chunks_file_path,
            candidate_count=candidate_count,
        )
    )

    keyword_candidates = (
        retrieve_keyword_candidates(
            query=query,
            chunks_file_path=chunks_file_path,
            candidate_count=(
                DEFAULT_KEYWORD_CANDIDATE_COUNT
            ),
        )
    )

    merged_candidates = merge_candidates(
        semantic_candidates=semantic_candidates,
        keyword_candidates=keyword_candidates,
    )

    expanded_candidates = add_neighbor_chunks(
        candidates=merged_candidates,
        chunks_file_path=chunks_file_path,
        neighbor_distance=DEFAULT_NEIGHBOR_DISTANCE,
    )

    ranked_results = rerank_results(
        query=query,
        candidates=expanded_candidates,
        top_k=top_k,
    )

    return add_relevant_page_tables(
        query=query,
        ranked_results=ranked_results,
        chunks_file_path=chunks_file_path,
    )
