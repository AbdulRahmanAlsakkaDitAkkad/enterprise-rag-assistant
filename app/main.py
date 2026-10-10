from fastapi import FastAPI

from app.api import chat, documents, health, processing, search


app = FastAPI(
    title="Enterprise RAG Assistant",
    description=(
        "A production-style RAG assistant for PDF question answering "
        "with citations."
    ),
    version="0.2.0",
)


app.include_router(health.router)
app.include_router(documents.router)
app.include_router(processing.router)
app.include_router(search.router)
app.include_router(chat.router)