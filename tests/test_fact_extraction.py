import json
from types import SimpleNamespace
from unittest.mock import patch

from app.services.llm_service import extract_supported_facts


def test_ambiguous_line_id_recovery_is_rejected():
    sentence = "The document explains German grammar."

    windows = [
        {
            "window_id": "window_1",
            "source_number": 1,
            "page_number": 1,
            "text": sentence,
        },
        {
            "window_id": "window_2",
            "source_number": 2,
            "page_number": 2,
            "text": sentence,
        },
    ]

    extraction_response = SimpleNamespace(
        message=SimpleNamespace(
            content=json.dumps({
                "facts": [
                    {
                        "window_id": "window_1",
                        "line_ids": ["invented_line_999"],
                        "fact": sentence,
                    }
                ]
            })
        )
    )

    with patch(
        "app.services.llm_service.client.chat",
        return_value=extraction_response,
    ), patch(
        "app.services.llm_service.verify_claim_against_lines"
    ) as verifier:
        result = extract_supported_facts(
            question="What does the document explain?",
            selected_windows=windows,
            conversation_history="",
        )

    assert result == []
    verifier.assert_not_called()


def test_unique_line_id_recovery_requires_verification():
    sentence = "The document explains German grammar."

    windows = [
        {
            "window_id": "window_1",
            "source_number": 1,
            "page_number": 1,
            "text": sentence,
        }
    ]

    extraction_response = SimpleNamespace(
        message=SimpleNamespace(
            content=json.dumps({
                "facts": [
                    {
                        "window_id": "window_1",
                        "line_ids": ["invented_line_999"],
                        "fact": sentence,
                    }
                ]
            })
        )
    )

    with patch(
        "app.services.llm_service.client.chat",
        return_value=extraction_response,
    ), patch(
        "app.services.llm_service.verify_claim_against_lines",
        return_value=False,
    ) as verifier:
        result = extract_supported_facts(
            question="What does the document explain?",
            selected_windows=windows,
            conversation_history="",
        )

    assert result == []
    verifier.assert_called_once()
    assert verifier.call_args.args[1] == sentence
    assert verifier.call_args.args[2] == sentence


def test_cross_window_citation_is_rejected():
    windows = [
        {
            "window_id": "window_1",
            "source_number": 1,
            "page_number": 1,
            "text": "The document explains German grammar.",
        },
        {
            "window_id": "window_2",
            "source_number": 2,
            "page_number": 2,
            "text": "The document explains English grammar.",
        },
    ]

    from app.services.llm_service import build_numbered_evidence_lines

    _, line_map = build_numbered_evidence_lines(windows)

    other_window_ids = [
        line_id
        for line_id, record in line_map.items()
        if record["window"]["window_id"] == "window_2"
    ]

    assert len(other_window_ids) == 1

    extraction_response = SimpleNamespace(
        message=SimpleNamespace(
            content=json.dumps({
                "facts": [
                    {
                        "window_id": "window_1",
                        "line_ids": [other_window_ids[0]],
                        "fact": "The document explains German grammar.",
                    }
                ]
            })
        )
    )

    with patch(
        "app.services.llm_service.client.chat",
        return_value=extraction_response,
    ), patch(
        "app.services.llm_service.verify_claim_against_lines"
    ) as verifier:
        result = extract_supported_facts(
            question="What does the document explain?",
            selected_windows=windows,
            conversation_history="",
        )

    assert result == []
    verifier.assert_not_called()


def test_valid_citation_does_not_bypass_semantic_verification():
    from app.services.llm_service import build_numbered_evidence_lines

    evidence = "The document explains German grammar."
    unsupported_fact = "The document explains quantum physics."

    windows = [
        {
            "window_id": "window_1",
            "source_number": 1,
            "page_number": 1,
            "text": evidence,
        }
    ]

    _, line_map = build_numbered_evidence_lines(windows)
    valid_line_id = next(iter(line_map))

    extraction_response = SimpleNamespace(
        message=SimpleNamespace(
            content=json.dumps({
                "facts": [
                    {
                        "window_id": "window_1",
                        "line_ids": [valid_line_id],
                        "fact": unsupported_fact,
                    }
                ]
            })
        )
    )

    with patch(
        "app.services.llm_service.client.chat",
        return_value=extraction_response,
    ), patch(
        "app.services.llm_service.verify_claim_against_lines",
        return_value=False,
    ) as verifier:
        result = extract_supported_facts(
            question="What does the document explain?",
            selected_windows=windows,
            conversation_history="",
        )

    assert result == []
    verifier.assert_called_once()
    assert verifier.call_args.args[1] == unsupported_fact
    assert verifier.call_args.args[2] == evidence
