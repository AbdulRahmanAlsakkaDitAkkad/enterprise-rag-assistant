from unittest.mock import patch

import pytest

from app.services.llm_service import verify_claim_against_lines


@pytest.mark.parametrize(
    "fact, evidence",
    [
        (
            'The document uses "wird".',
            "The document uses wurde.",
        ),
        (
            'The example contains "konnte".',
            "The example contains k\u00f6nnte.",
        ),
        (
            "The construction is wird + Infinitiv.",
            "The construction is w\u00fcrde + Infinitiv.",
        ),
    ],
)
def test_unsupported_forms_rejected_without_llm(fact, evidence):
    with patch("app.services.llm_service.client.chat") as chat_mock:
        result = verify_claim_against_lines(
            question="What grammatical form is used?",
            fact=fact,
            evidence=evidence,
        )

    assert result is False
    chat_mock.assert_not_called()


def test_invalid_verifier_response_rejects_claim():
    from types import SimpleNamespace

    invalid_response = SimpleNamespace(
        message=SimpleNamespace(
            content="This is not valid JSON."
        )
    )

    with patch(
        "app.services.llm_service.client.chat",
        return_value=invalid_response,
    ) as chat_mock:
        result = verify_claim_against_lines(
            question="What does the document explain?",
            fact="The document describes a grammar rule.",
            evidence="The document describes a grammar rule.",
        )

    assert result is False
    chat_mock.assert_called_once()


def test_semantic_verifier_rejects_unsupported_claim():
    from types import SimpleNamespace
    import json

    response = SimpleNamespace(
        message=SimpleNamespace(
            content=json.dumps(
                {
                    "supported": False,
                    "reason": "The cited evidence does not support the claim.",
                }
            )
        )
    )

    with patch(
        "app.services.llm_service.client.chat",
        return_value=response,
    ) as chat_mock:
        result = verify_claim_against_lines(
            question="What does the document explain?",
            fact="The document describes quantum physics.",
            evidence="The document explains German grammar.",
        )

    assert result is False
    chat_mock.assert_called_once()


def test_semantic_verifier_accepts_supported_claim():
    from types import SimpleNamespace
    import json

    response = SimpleNamespace(
        message=SimpleNamespace(
            content=json.dumps(
                {
                    "supported": True,
                    "reason": "The claim is directly stated in the evidence.",
                }
            )
        )
    )

    with patch(
        "app.services.llm_service.client.chat",
        return_value=response,
    ) as chat_mock:
        result = verify_claim_against_lines(
            question="What does the document explain?",
            fact="The document explains German grammar.",
            evidence="The document explains German grammar.",
        )

    assert result is True
    chat_mock.assert_called_once()
