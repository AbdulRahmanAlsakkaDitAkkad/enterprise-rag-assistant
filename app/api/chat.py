from pathlib import Path

from fastapi import (
    APIRouter,
    HTTPException,
)

from app.schemas.chat import (
    MultiRAGRequest,
    MultiRAGResponse,
    RAGRequest,
    RAGResponse,
)

from app.services.chat_memory import (
    add_message,
    create_session,
    get_recent_messages,
    session_exists,
)

from app.services.document_registry import (
    get_document,
)

from app.services.faiss_service import (
    rerank_results,
    retrieve_faiss_candidates,
    search_faiss_index,
)

from app.services.llm_service import (
    generate_rag_answer,
)

from app.services.query_decomposer import (
    decompose_query,
)

from app.services.query_rewriter import (
    rewrite_query,
)


router = APIRouter(
    prefix="/chat",
    tags=["Chat"],
)


def resolve_session(
    requested_session_id: str | None,
) -> str:
    """
    Reuse an existing session or create a new one.

    If the client provides a session ID that does not exist,
    return a 404 error instead of silently creating another session.
    """
    if requested_session_id is None:
        return create_session()

    if not session_exists(
        requested_session_id
    ):
        raise HTTPException(
            status_code=404,
            detail="Chat session not found.",
        )

    return requested_session_id


def retrieve_decomposed_queries(
    retrieval_query: str,
    index_path: Path,
    chunks_path: Path,
    top_k: int,
) -> list[dict]:
    """
    Retrieve evidence using query decomposition.

    Strategy:
    1. Decompose comparison questions into focused queries.
    2. Retrieve results independently for every query.
    3. Preserve the strongest unique result from each focused
       subquery so one concept cannot eliminate another.
    4. Merge all remaining candidates.
    5. Globally rerank the remaining candidates against the
       original standalone query.
    6. Return at most top_k chunks.
    """
    queries = decompose_query(
        retrieval_query
    )

    # Normal question: no decomposition was necessary.
    if len(queries) == 1:
        return search_faiss_index(
            query=retrieval_query,
            index_path=index_path,
            chunks_file_path=chunks_path,
            top_k=top_k,
        )

    per_query_top_k = max(
        top_k,
        5,
    )

    query_results = []

    for query in queries:
        results = search_faiss_index(
            query=query,
            index_path=index_path,
            chunks_file_path=chunks_path,
            top_k=per_query_top_k,
        )

        query_results.append(
            {
                "query": query,
                "results": results,
            }
        )

    selected_results = []
    selected_chunk_ids = set()

    # queries[0] is the original question.
    # queries[1:] are focused concept queries.
    #
    # First guarantee one result from each focused query.
    for query_group in query_results[1:]:
        for result in query_group["results"]:
            chunk_id = result["chunk_id"]

            if chunk_id in selected_chunk_ids:
                continue

            selected_result = result.copy()

            selected_result[
                "matched_query"
            ] = query_group["query"]

            selected_results.append(
                selected_result
            )

            selected_chunk_ids.add(
                chunk_id
            )

            break

    # Collect every remaining unique candidate.
    remaining_candidates = {}

    for query_group in query_results:
        for result in query_group["results"]:
            chunk_id = result["chunk_id"]

            if chunk_id in selected_chunk_ids:
                continue

            existing = remaining_candidates.get(
                chunk_id
            )

            if existing is None:
                candidate = result.copy()

                candidate[
                    "matched_query"
                ] = query_group["query"]

                remaining_candidates[
                    chunk_id
                ] = candidate

                continue

            existing_score = existing.get(
                "reranker_score",
                float("-inf"),
            )

            new_score = result.get(
                "reranker_score",
                float("-inf"),
            )

            if new_score > existing_score:
                candidate = result.copy()

                candidate[
                    "matched_query"
                ] = query_group["query"]

                remaining_candidates[
                    chunk_id
                ] = candidate

    remaining_slots = max(
        top_k - len(selected_results),
        0,
    )

    if (
        remaining_candidates
        and remaining_slots > 0
    ):
        globally_ranked = rerank_results(
            query=retrieval_query,
            candidates=list(
                remaining_candidates.values()
            ),
            top_k=remaining_slots,
        )

        selected_results.extend(
            globally_ranked
        )

    # Rerank the final evidence set against the original
    # standalone question for consistent final ordering.
    final_results = rerank_results(
        query=retrieval_query,
        candidates=selected_results,
        top_k=min(
            top_k,
            len(selected_results),
        ),
    )

    return final_results



@router.post(
    "",
    response_model=RAGResponse,
)
def rag_chat(
    request: RAGRequest,
):
    """
    Ask a question about one processed document.

    Conversation history is used to understand follow-up questions,
    while retrieved document chunks remain the factual source.
    """
    try:
        # 1. Create or reuse conversation session.
        session_id = resolve_session(
            request.session_id
        )

        # 2. Load previous conversation before adding
        # the current question.
        chat_history = get_recent_messages(
            session_id=session_id,
            limit=6,
        )

        # 3. Resolve document.
        document = get_document(
            request.document_id
        )

        if document is None:
            raise HTTPException(
                status_code=404,
                detail="Document not found.",
            )

        chunks_path = Path(
            document["chunks_file_path"]
        )

        index_path = Path(
            document["faiss_index_path"]
        )

        if not chunks_path.exists():
            raise HTTPException(
                status_code=500,
                detail=(
                    "The document chunks file is missing."
                ),
            )

        if not index_path.exists():
            raise HTTPException(
                status_code=500,
                detail=(
                    "The FAISS index file is missing."
                ),
            )

        # 4. Rewrite conversational follow-ups into a
        # standalone retrieval query.
        retrieval_query = rewrite_query(
            question=request.question,
            chat_history=chat_history,
        )

        # 5. Decompose comparison/multi-concept queries,
        # retrieve each independently, merge them and
        # rerank against the original standalone query.
        retrieved_chunks = (
            retrieve_decomposed_queries(
                retrieval_query=(
                    retrieval_query
                ),
                index_path=index_path,
                chunks_path=chunks_path,
                top_k=request.top_k,
            )
        )

        # 6. Generate grounded answer.
        answer = generate_rag_answer(
            question=retrieval_query,
            retrieved_chunks=(
                retrieved_chunks
            ),
            chat_history=chat_history,
        )

        # 7. Store successful conversation turn.
        add_message(
            session_id=session_id,
            role="user",
            content=request.question,
        )

        add_message(
            session_id=session_id,
            role="assistant",
            content=answer,
        )

        # 8. Build API source response.
        # Preserve retrieval source numbers so they match
        # the backend-controlled citations in the answer.
        import re

        cited_source_numbers = {
            int(number)
            for number in re.findall(
                r"\[Source\s+(\d+)\s*,\s*page\s+\d+\]",
                answer,
                flags=re.IGNORECASE,
            )
        }

        sources = [
            {
                "source_number": (
                    source_number
                ),
                "source_file": (
                    result["source_file"]
                ),
                "page_number": (
                    result["page_number"]
                ),
                "score": result["score"],
                "reranker_score": (
                    result.get(
                        "reranker_score"
                    )
                ),
                "text": result["text"],
            }
            for source_number, result
            in enumerate(
                retrieved_chunks,
                start=1,
            )
            if source_number in cited_source_numbers
        ]

        return {
            "session_id": session_id,
            "document_id": (
                request.document_id
            ),
            "question": request.question,
            "answer": answer,
            "sources": sources,
        }

    except HTTPException:
        raise

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=(
                "RAG answer generation "
                f"failed: {error}"
            ),
        ) from error


def retrieve_best_from_document(
    document_id: str,
    question: str,
    results_per_document: int,
) -> list[dict]:
    """
    Retrieve a large candidate pool from one document
    and rerank it independently.
    """
    document = get_document(
        document_id
    )

    if document is None:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Document not found: "
                f"{document_id}"
            ),
        )

    chunks_path = Path(
        document["chunks_file_path"]
    )

    index_path = Path(
        document["faiss_index_path"]
    )

    if not chunks_path.exists():
        raise HTTPException(
            status_code=500,
            detail=(
                "Chunks file missing for "
                f"document: {document_id}"
            ),
        )

    if not index_path.exists():
        raise HTTPException(
            status_code=500,
            detail=(
                "FAISS index missing for "
                f"document: {document_id}"
            ),
        )

    candidates = retrieve_faiss_candidates(
        query=question,
        index_path=index_path,
        chunks_file_path=chunks_path,
        candidate_count=100,
    )

    best_results = rerank_results(
        query=question,
        candidates=candidates,
        top_k=results_per_document,
    )

    for result in best_results:
        result["document_id"] = (
            document_id
        )

    return best_results


def build_diverse_multi_results(
    question: str,
    document_ids: list[str],
    top_k: int,
) -> list[dict]:
    """
    Build a diverse retrieval result set across all documents.
    """
    results_per_document = max(
        3,
        top_k,
    )

    per_document_results = []

    for document_id in document_ids:
        document_results = (
            retrieve_best_from_document(
                document_id=document_id,
                question=question,
                results_per_document=(
                    results_per_document
                ),
            )
        )

        if document_results:
            per_document_results.append(
                document_results
            )

    if not per_document_results:
        return []

    selected_results = []
    selected_chunk_ids = set()

    # Guarantee one strong result from each document.
    for document_results in (
        per_document_results
    ):
        best_result = document_results[0]

        chunk_id = best_result["chunk_id"]

        if chunk_id not in selected_chunk_ids:
            selected_results.append(
                best_result
            )

            selected_chunk_ids.add(
                chunk_id
            )

    remaining_candidates = []

    for document_results in (
        per_document_results
    ):
        for result in document_results[1:]:
            if (
                result["chunk_id"]
                not in selected_chunk_ids
            ):
                remaining_candidates.append(
                    result
                )

    remaining_slots = max(
        top_k - len(selected_results),
        0,
    )

    if (
        remaining_candidates
        and remaining_slots > 0
    ):
        globally_ranked = rerank_results(
            query=question,
            candidates=(
                remaining_candidates
            ),
            top_k=remaining_slots,
        )

        selected_results.extend(
            globally_ranked
        )

    selected_results.sort(
        key=lambda item: item.get(
            "reranker_score",
            float("-inf"),
        ),
        reverse=True,
    )

    return selected_results[:top_k]


def select_llm_evidence(
    retrieved_results: list[dict],
) -> list[dict]:
    """
    Select only strong evidence for the LLM.

    Strategy:
    - Sort by reranker score.
    - Keep results sufficiently close to the best result.
    - Preserve document diversity.
    - Keep at most 3 chunks.
    """
    if not retrieved_results:
        return []

    sorted_results = sorted(
        retrieved_results,
        key=lambda item: item.get(
            "reranker_score",
            float("-inf"),
        ),
        reverse=True,
    )

    best_score = sorted_results[0].get(
        "reranker_score",
        float("-inf"),
    )

    relative_margin = 1.6

    strong_results = [
        result
        for result in sorted_results
        if result.get(
            "reranker_score",
            float("-inf"),
        ) >= best_score - relative_margin
    ]

    selected = []
    used_documents = set()
    used_chunks = set()

    # First keep one strong result
    # from each document.
    for result in strong_results:
        document_id = result.get(
            "document_id"
        )

        chunk_id = result["chunk_id"]

        if document_id in used_documents:
            continue

        if chunk_id in used_chunks:
            continue

        selected.append(
            result
        )

        used_documents.add(
            document_id
        )

        used_chunks.add(
            chunk_id
        )

        if len(selected) >= 3:
            return selected

    # Fill remaining slots with
    # strong unused results.
    for result in strong_results:
        chunk_id = result["chunk_id"]

        if chunk_id in used_chunks:
            continue

        selected.append(
            result
        )

        used_chunks.add(
            chunk_id
        )

        if len(selected) >= 3:
            break

    return selected


@router.post(
    "/multi",
    response_model=MultiRAGResponse,
)
def multi_rag_chat(
    request: MultiRAGRequest,
):
    """
    Ask a question across multiple processed documents
    while maintaining conversation memory.
    """
    try:
        # 1. Create or reuse conversation session.
        session_id = resolve_session(
            request.session_id
        )

        # 2. Load previous conversation.
        chat_history = get_recent_messages(
            session_id=session_id,
            limit=30,
        )

        # 3. Rewrite the question before
        # searching across documents.
        retrieval_query = rewrite_query(
            question=request.question,
            chat_history=chat_history,
        )

        # 4. Retrieve diverse evidence
        # across documents.
        final_results = (
            build_diverse_multi_results(
                question=retrieval_query,
                document_ids=(
                    request.document_ids
                ),
                top_k=request.top_k,
            )
        )

        # 5. Keep only strong evidence
        # for generation.
        llm_evidence = select_llm_evidence(
            final_results
        )

        # 6. Generate grounded answer
        # with conversation memory.
        answer = generate_rag_answer(
            question=retrieval_query,
            retrieved_chunks=llm_evidence,
            chat_history=chat_history,
        )

        # 7. Store successful conversation turn.
        add_message(
            session_id=session_id,
            role="user",
            content=request.question,
        )

        add_message(
            session_id=session_id,
            role="assistant",
            content=answer,
        )

        # 8. Build response sources.
        sources = [
            {
                "source_number": (
                    source_number
                ),
                "source_file": (
                    result["source_file"]
                ),
                "page_number": (
                    result["page_number"]
                ),
                "score": result["score"],
                "reranker_score": (
                    result.get(
                        "reranker_score"
                    )
                ),
                "text": result["text"],
            }
            for source_number, result
            in enumerate(
                llm_evidence,
                start=1,
            )
        ]

        # Return only sources cited in the final answer.
        # Preserve original retrieval source numbers.
        import re

        cited_source_numbers = {
            int(number)
            for number in re.findall(
                r"\[Source\s+(\d+)\s*,\s*page\s+\d+\]",
                answer,
                flags=re.IGNORECASE,
            )
        }

        sources = [
            source
            for source in sources
            if source["source_number"] in cited_source_numbers
        ]

        return {
            "session_id": session_id,
            "document_ids": (
                request.document_ids
            ),
            "question": request.question,
            "answer": answer,
            "sources": sources,
        }

    except HTTPException:
        raise

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=(
                "Multi-document RAG "
                f"failed: {error}"
            ),
        ) from error