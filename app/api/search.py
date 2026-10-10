from pathlib import Path

from fastapi import APIRouter, HTTPException

from app.services.faiss_service import search_faiss_index


router = APIRouter(
    prefix="/search",
    tags=["Search"],
)


@router.post("/semantic")
def semantic_search(
    query: str,
    chunks_file_path: str,
    faiss_index_path: str,
    top_k: int = 5,
):
    try:
        chunks_path = Path(chunks_file_path)
        index_path = Path(faiss_index_path)

        if not chunks_path.exists():
            raise HTTPException(
                status_code=404,
                detail="Chunks file not found.",
            )

        if not index_path.exists():
            raise HTTPException(
                status_code=404,
                detail="FAISS index file not found.",
            )

        results = search_faiss_index(
            query=query,
            index_path=index_path,
            chunks_file_path=chunks_path,
            top_k=top_k,
        )

        return {
            "message": "Semantic search completed successfully",
            "query": query,
            "top_k": top_k,
            "results": results,
        }

    except HTTPException:
        raise

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Semantic search failed: {error}",
        ) from error