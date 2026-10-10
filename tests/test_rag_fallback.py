import pytest

from app.services.llm_service import (
    FALLBACK_ANSWER_DE,
    FALLBACK_ANSWER_EN,
    generate_rag_answer,
)


def test_english_fallback_when_no_evidence():
    answer = generate_rag_answer(
        question="What does the document explain?",
        retrieved_chunks=[],
    )

    assert answer == FALLBACK_ANSWER_EN


def test_german_fallback_when_no_evidence():
    answer = generate_rag_answer(
        question="Was steht in dem Dokument?",
        retrieved_chunks=[],
    )

    assert answer == FALLBACK_ANSWER_DE


def test_empty_question_is_rejected():
    with pytest.raises(
        ValueError,
        match="The question cannot be empty",
    ):
        generate_rag_answer(
            question="   ",
            retrieved_chunks=[],
        )


def test_fallback_when_no_windows_selected():
    from unittest.mock import patch

    retrieved_chunks = [
        {
            "chunk_id": "chunk_1",
            "text": "Unrelated document content.",
            "page_number": 1,
        }
    ]

    with (
        patch(
            "app.services.llm_service.select_windows_for_question",
            return_value=[],
        ) as select_mock,
        patch(
            "app.services.llm_service.generate_answer_from_windows",
        ) as generate_mock,
    ):
        answer = generate_rag_answer(
            question="What does the document say about quantum physics?",
            retrieved_chunks=retrieved_chunks,
        )

    assert answer == FALLBACK_ANSWER_EN
    select_mock.assert_called_once()
    generate_mock.assert_not_called()


def test_fallback_when_generated_answer_is_empty():
    from types import SimpleNamespace
    from unittest.mock import patch

    selected_windows = [
        {
            "window_id": "window_1",
            "source_number": 1,
            "page_number": 1,
            "text": "Some document evidence.",
        }
    ]

    with (
        patch(
            "app.services.llm_service.select_windows_for_question",
            return_value=selected_windows,
        ) as select_mock,
        patch(
            "app.services.llm_service.generate_answer_from_windows",
            return_value=SimpleNamespace(answer="   "),
        ) as generate_mock,
        patch(
            "app.services.llm_service.append_citations_from_windows",
        ) as citation_mock,
    ):
        answer = generate_rag_answer(
            question="What does the document explain?",
            retrieved_chunks=[{"text": "Some document evidence."}],
        )

    assert answer == FALLBACK_ANSWER_EN
    select_mock.assert_called_once()
    generate_mock.assert_called_once()
    citation_mock.assert_not_called()


def test_fallback_when_citation_processing_returns_empty():
    from types import SimpleNamespace
    from unittest.mock import patch

    selected_windows = [
        {
            "window_id": "window_1",
            "source_number": 1,
            "page_number": 1,
            "text": "Some document evidence.",
        }
    ]

    with (
        patch(
            "app.services.llm_service.select_windows_for_question",
            return_value=selected_windows,
        ) as select_mock,
        patch(
            "app.services.llm_service.generate_answer_from_windows",
            return_value=SimpleNamespace(
                answer="A generated answer."
            ),
        ) as generate_mock,
        patch(
            "app.services.llm_service.append_citations_from_windows",
            return_value="",
        ) as citation_mock,
    ):
        answer = generate_rag_answer(
            question="What does the document explain?",
            retrieved_chunks=[{"text": "Some document evidence."}],
        )

    assert answer == FALLBACK_ANSWER_EN
    select_mock.assert_called_once()
    generate_mock.assert_called_once()
    citation_mock.assert_called_once()
