from fastapi import APIRouter


router = APIRouter(tags=["Health"])


@router.get("/health")
def health_check():
    return {
        "status": "ok",
        "message": "Enterprise RAG Assistant API is running",
        "version": "0.2.0",
    }