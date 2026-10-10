import re


def citations_from_answer(answer):
    return {
        (int(source), int(page))
        for source, page in re.findall(
            r"\[Source\s+(\d+)\s*,\s*page\s+(\d+)\]",
            answer,
            flags=re.IGNORECASE,
        )
    }


def sources_from_response(sources):
    return {
        (item["source_number"], item["page_number"])
        for item in sources
    }


def citations_are_aligned(answer, sources):
    citations = citations_from_answer(answer)
    returned = sources_from_response(sources)
    return bool(citations) and citations == returned


def test_correct_citation_alignment():
    answer = (
        "Entschuldigung, k\u00f6nnten Sie mir bitte helfen? "
        "[Source 1, page 133]"
    )
    sources = [
        {"source_number": 1, "page_number": 133}
    ]

    assert citations_are_aligned(answer, sources)


def test_wrong_page_is_rejected():
    answer = "Example. [Source 1, page 133]"
    sources = [
        {"source_number": 1, "page_number": 134}
    ]

    assert not citations_are_aligned(answer, sources)


def test_missing_citation_is_rejected():
    answer = "An answer without a citation."
    sources = [
        {"source_number": 1, "page_number": 133}
    ]

    assert not citations_are_aligned(answer, sources)


def test_extra_source_is_rejected():
    answer = "Example. [Source 1, page 133]"
    sources = [
        {"source_number": 1, "page_number": 133},
        {"source_number": 2, "page_number": 134},
    ]

    assert not citations_are_aligned(answer, sources)


def test_multiple_correct_citations():
    answer = (
        "First fact. [Source 1, page 133] "
        "Second fact. [Source 2, page 134]"
    )
    sources = [
        {"source_number": 1, "page_number": 133},
        {"source_number": 2, "page_number": 134},
    ]

    assert citations_are_aligned(answer, sources)
