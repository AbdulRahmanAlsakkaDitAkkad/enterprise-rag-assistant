from pydantic import BaseModel, Field


class RAGRequest(BaseModel):
    document_id: str = Field(
        min_length=5,
        description="Internal ID of the processed document.",
    )

    question: str = Field(
        min_length=2,
        description="Question to answer from the document.",
    )

    session_id: str | None = Field(
        default=None,
        description=(
            "Optional chat session ID. "
            "If omitted, a new session will be created."
        ),
    )

    top_k: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Maximum number of retrieved source chunks.",
    )


class MultiRAGRequest(BaseModel):
    document_ids: list[str] = Field(
        min_length=2,
        description=(
            "Internal IDs of the processed documents "
            "to search together."
        ),
    )

    question: str = Field(
        min_length=2,
        description=(
            "Question to answer using all selected documents."
        ),
    )

    session_id: str | None = Field(
        default=None,
        description=(
            "Optional chat session ID. "
            "If omitted, a new session will be created."
        ),
    )

    top_k: int = Field(
        default=5,
        ge=1,
        le=10,
        description=(
            "Maximum number of final retrieved source chunks."
        ),
    )


class SourceResponse(BaseModel):
    source_number: int
    source_file: str
    page_number: int

    score: float

    reranker_score: float | None = None

    text: str


class RAGResponse(BaseModel):
    session_id: str
    document_id: str
    question: str
    answer: str
    sources: list[SourceResponse]


class MultiRAGResponse(BaseModel):
    session_id: str
    document_ids: list[str]
    question: str
    answer: str
    sources: list[SourceResponse]