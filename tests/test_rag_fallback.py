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
