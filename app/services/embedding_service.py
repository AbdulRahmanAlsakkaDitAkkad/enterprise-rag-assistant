import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


MODEL_NAME = (
    "sentence-transformers/"
    "paraphrase-multilingual-MiniLM-L12-v2"
)
EMBEDDINGS_DIR = Path("data/indexes")


model = SentenceTransformer(MODEL_NAME)


def load_chunks(chunks_file_path: Path) -> list[dict]:
    with open(chunks_file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return data["chunks"]


def create_embeddings_for_chunks(chunks: list[dict]) -> np.ndarray:
    texts = [chunk["text"] for chunk in chunks]

    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    return embeddings


def save_embeddings(chunks_file_path: Path, embeddings: np.ndarray) -> Path:
    output_path = EMBEDDINGS_DIR / f"{chunks_file_path.stem}_embeddings.npy"

    np.save(output_path, embeddings)

    return output_path