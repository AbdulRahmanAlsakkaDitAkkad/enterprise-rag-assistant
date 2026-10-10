import pytest

from app.services.llm_service import (
    extract_verified_table_cells,
    match_question_table_cells,
)


@pytest.mark.parametrize(
    "table_text",
    [
        "Person | koennen | sollen\\nich | koennte",
        "Person | koennen | koennen\\nich | koennte | sollte",
        "Person | koennen | sollen\\nich | koennte |",
    ],
)
def test_malformed_tables_are_rejected(table_text):
    windows = [
        {
            "window_id": "window_1",
            "source_number": 1,
            "page_number": 1,
            "text": table_text,
        }
    ]

    assert extract_verified_table_cells(windows) == []

    assert match_question_table_cells(
        question="Konjunktiv II von koennen und sollen",
        selected_windows=windows,
    ) == []


def test_valid_table_cells_preserve_values_and_provenance():
    table_text = (
        "Person | koennen | sollen\n"
        "ich | koennte | sollte\n"
        "du | koenntest | solltest"
    )

    windows = [
        {
            "window_id": "window_1",
            "source_number": 1,
            "page_number": 1,
            "text": table_text,
        }
    ]

    cells = extract_verified_table_cells(windows)

    assert len(cells) == 4

    values = {
        (cell["row_label"], cell["header"]): cell["value"]
        for cell in cells
    }

    assert values == {
        ("ich", "koennen"): "koennte",
        ("ich", "sollen"): "sollte",
        ("du", "koennen"): "koenntest",
        ("du", "sollen"): "solltest",
    }

    for cell in cells:
        assert cell["window_id"] == "window_1"
        assert len(cell["line_ids"]) == 2
        assert cell["line_ids"][0] == "window_1_line_1"
        assert cell["value"] in cell["evidence"]

    matched = match_question_table_cells(
        question="Konjunktiv II von koennen und sollen",
        selected_windows=windows,
    )

    assert len(matched) == 4


def test_deterministic_table_answer_uses_literal_values():
    from unittest.mock import patch
    from app.services.llm_service import generate_answer_from_windows

    windows = [
        {
            "window_id": "table_window",
            "source_number": 1,
            "page_number": 10,
            "text": (
                "Person | koennen | sollen\n"
                "ich | koennte | sollte\n"
                "du | koenntest | solltest"
            ),
        },
        {
            "window_id": "context_window",
            "source_number": 1,
            "page_number": 10,
            "text": "Konjunktiv 2",
        },
    ]

    with patch(
        "app.services.llm_service.extract_supported_facts"
    ) as extraction_mock:
        result = generate_answer_from_windows(
            question="Konjunktiv II von koennen und sollen",
            selected_windows=windows,
            conversation_history="",
        )

    assert result.answer == (
        "Person | koennen | sollen\n"
        "ich | koennte | sollte\n"
        "du | koenntest | solltest"
    )

    extraction_mock.assert_not_called()
    assert len(windows) == 1
    assert windows[0]["window_id"] == "table_window"


def test_malformed_table_cannot_trigger_deterministic_answer():
    from unittest.mock import patch
    from app.services.llm_service import generate_answer_from_windows

    windows = [
        {
            "window_id": "table_window",
            "source_number": 1,
            "page_number": 10,
            "text": (
                "Person | koennen | koennen\n"
                "ich | koennte | sollte"
            ),
        },
        {
            "window_id": "context_window",
            "source_number": 1,
            "page_number": 10,
            "text": "Konjunktiv 2",
        },
    ]

    with patch(
        "app.services.llm_service.extract_supported_facts",
        return_value=[],
    ) as extraction_mock:
        result = generate_answer_from_windows(
            question="Konjunktiv II von koennen und sollen",
            selected_windows=windows,
            conversation_history="",
        )

    assert result.answer == ""
    extraction_mock.assert_called_once()
    assert windows == []
