
import re


COMPARISON_PATTERNS = [
    r"\bunterschied zwischen\s+(.+?)\s+und\s+(.+?)(?:\?|$)",
    r"\bunterscheiden sich\s+(.+?)\s+und\s+(.+?)(?:\?|$)",
    r"\bcompare\s+(.+?)\s+(?:and|with|to)\s+(.+?)(?:\?|$)",
    r"\bdifference between\s+(.+?)\s+and\s+(.+?)(?:\?|$)",
]


def clean_concept(concept: str) -> str:
    concept = concept.strip()
    concept = re.sub(r"[?.!,;:]+$", "", concept)
    return concept.strip()


def extract_comparison_concepts(question: str) -> list[str]:
    cleaned_question = question.strip()

    for pattern in COMPARISON_PATTERNS:
        match = re.search(
            pattern,
            cleaned_question,
            flags=re.IGNORECASE,
        )

        if not match:
            continue

        first = clean_concept(match.group(1))
        second = clean_concept(match.group(2))

        if first and second and first.lower() != second.lower():
            return [first, second]

    return []


def detect_german(question: str) -> bool:
    words = set(
        re.findall(r"\b[\wäöüß]+\b", question.lower())
    )

    german_markers = {
        "was", "wie", "warum", "welche", "welcher",
        "welches", "sind", "ist", "unterschied",
        "unterscheiden", "zwischen", "und",
        "der", "die", "das",
    }

    return bool(words.intersection(german_markers))


def build_concept_query(concept: str, german: bool) -> str:
    if german:
        return (
            f"{concept} Bedeutung Gebrauch "
            f"Erklärung Regel Grammatik"
        )

    return (
        f"{concept} meaning usage "
        f"explanation rule grammar"
    )


def extract_primary_subject(question: str) -> str | None:
    patterns = [
        r"\bwas bedeutet\s+([\wäöüß-]+)",
        r"\bwas heißt\s+([\wäöüß-]+)",
        r"\bwhat does\s+([\w-]+)\s+mean",
        r"\bmeaning of\s+([\w-]+)",
    ]

    for pattern in patterns:
        match = re.search(
            pattern,
            question,
            flags=re.IGNORECASE,
        )

        if match:
            return match.group(1)

    return None


def split_compound_question(question: str) -> list[str]:
    """
    Split independent questions joined by 'und' or 'and'.

    Only split when the second part begins with a recognizable
    question structure. This avoids splitting noun phrases
    such as 'können und sollen'.
    """
    patterns = [
        r"\s+und\s+(?=(?:ist|sind|warum|wie|was|wann|wo|kann|darf|muss|soll)\b)",
        r"\s+and\s+(?=(?:is|are|why|how|what|when|where|can|does|do)\b)",
    ]

    parts = [question]

    for pattern in patterns:
        next_parts = []

        for part in parts:
            next_parts.extend(
                re.split(pattern, part, flags=re.IGNORECASE)
            )

        parts = next_parts

    return [
        part.strip().rstrip("?").strip()
        for part in parts
        if part.strip().rstrip("?").strip()
    ]


def decompose_query(question: str) -> list[str]:
    cleaned_question = question.strip()

    if not cleaned_question:
        raise ValueError("The question cannot be empty.")

    queries = [cleaned_question]
    german = detect_german(cleaned_question)

    # Preserve existing comparison behavior.
    concepts = extract_comparison_concepts(cleaned_question)

    if concepts:
        for concept in concepts:
            focused_query = build_concept_query(
                concept=concept,
                german=german,
            )

            if focused_query not in queries:
                queries.append(focused_query)

        return queries

    # Handle compound questions.
    parts = split_compound_question(cleaned_question)

    if len(parts) > 1:
        subject = extract_primary_subject(parts[0])

        if subject:
            focused_query = build_concept_query(
                concept=subject,
                german=german,
            )

            if focused_query not in queries:
                queries.append(focused_query)

        for part in parts[1:]:
            focused_part = part

            if subject:
                focused_part = f"{subject} {part}"

            if focused_part not in queries:
                queries.append(focused_part)

    return queries
