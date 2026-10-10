from pathlib import Path

from fastapi import APIRouter, HTTPException

from app.services.chunking_service import (
    create_chunks_from_extracted_json,
    save_chunks,
)
from app.services.embedding_service import (
    create_embeddings_for_chunks,
    load_chunks,
    save_embeddings,
)
from app.services.faiss_service import (
    create_faiss_index,
    load_embeddings,
    save_faiss_index,
)


router = APIRouter(
    prefix="/processing",
    tags=["Processing"],
)


@router.post("/chunks")
def chunk_document(extracted_file_path: str):
    try:
        extracted_path = Path(extracted_file_path)

        if not extracted_path.exists():
            raise HTTPException(
                status_code=404,
                detail="Extracted file not found.",
            )

        chunks = create_chunks_from_extracted_json(extracted_path)
        chunks_path = save_chunks(extracted_path, chunks)

        return {
            "message": "Document chunked successfully",
            "source_file": extracted_file_path,
            "chunks_count": len(chunks),
            "chunks_file": str(chunks_path),
            "preview": chunks[0] if chunks else None,
        }

    except HTTPException:
        raise

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Chunking failed: {error}",
        ) from error


@router.post("/embeddings")
def create_embeddings(chunks_file_path: str):
    try:
        chunks_path = Path(chunks_file_path)

        if not chunks_path.exists():
            raise HTTPException(
                status_code=404,
                detail="Chunks file not found.",
            )

        chunks = load_chunks(chunks_path)
        embeddings = create_embeddings_for_chunks(chunks)
        embeddings_path = save_embeddings(chunks_path, embeddings)

        return {
            "message": "Embeddings created successfully",
            "chunks_file": chunks_file_path,
            "chunks_count": len(chunks),
            "embedding_model": (
                "sentence-transformers/all-MiniLM-L6-v2"
            ),
            "embedding_shape": list(embeddings.shape),
            "embeddings_file": str(embeddings_path),
        }

    except HTTPException:
        raise

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Embedding creation failed: {error}",
        ) from error


@router.post("/faiss-index")
def build_faiss_index(
    chunks_file_path: str,
    embeddings_file_path: str,
):
    try:
        chunks_path = Path(chunks_file_path)
        embeddings_path = Path(embeddings_file_path)

        if not chunks_path.exists():
            raise HTTPException(
                status_code=404,
                detail="Chunks file not found.",
            )

        if not embeddings_path.exists():
            raise HTTPException(
                status_code=404,
                detail="Embeddings file not found.",
            )

        embeddings = load_embeddings(embeddings_path)
        index = create_faiss_index(embeddings)
        index_path = save_faiss_index(index, chunks_path)

        return {
            "message": "FAISS index created successfully",
            "number_of_vectors": index.ntotal,
            "embedding_dimension": embeddings.shape[1],
            "faiss_index_file": str(index_path),
        }

    except HTTPException:
        raise

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"FAISS index creation failed: {error}",
        ) from error