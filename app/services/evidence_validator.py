import re
import unicodedata
from difflib import SequenceMatcher

from ollama import Client
from pydantic import BaseModel, Field, ValidationError


MIN_EVIDENCE_SIMILARITY = 0.72

OLLAMA_HOST = "http://127.0.0.1:11434"
MODEL_NAME = "qwen2.5:7b"


OCR_REPLACEMENTS = {
    "6": "o",
    "0": "o",
    "é": "e",
    "è": "e",
    "ê": "e",
    "ii": "u",
    "iii": "u",
}


client = Client(
    host=OLLAMA_HOST
)


class EntailmentDecision(BaseModel):
    supported: bool = Field(
        description=(
            "True only if the evidence directly supports "
            "the entire claim."
        )
    )

    reason: str = Field(
        description=(
            "Short explanation of why the evidence does "
            "or does not support the claim."
        )
    )


def normalize_text(
    text: str,
) -> str:
    """
    Normalize OCR-heavy text before comparing evidence.

    This normalization is used only to verify whether
    evidence occurs in the referenced source.
    """
    normalized = text.lower()

    normalized = unicodedata.normalize(
        "NFKD",
        normalized,
    )

    normalized = "".join(
        character
        for character in normalized
        if not unicodedata.combining(
            character
        )
    )

    for old, new in OCR_REPLACEMENTS.items():
        normalized = normalized.replace(
            old,
            new,
        )

    normalized = re.sub(
        r"[^a-z0-9äöüß ]+",
        " ",
        normalized,
    )

    normalized = re.sub(
        r"\s+",
        " ",
        normalized,
    ).strip()

    return normalized


def calculate_similarity(
    text_a: str,
    text_b: str,
) -> float:
    """
    Calculate approximate similarity between two
    normalized strings.
    """
    return SequenceMatcher(
        None,
        text_a,
        text_b,
    ).ratio()


def evidence_exists_in_source(
    evidence: str,
    source_text: str,
) -> bool:
    """
    Verify that evidence occurs in the referenced source.

    Strategy:
    1. Normalize OCR differences.
    2. Try exact normalized containment.
    3. Try conservative fuzzy matching.
    """
    normalized_evidence = normalize_text(
        evidence
    )

    normalized_source = normalize_text(
        source_text
    )

    if not normalized_evidence:
        return False

    if not normalized_source:
        return False

    if normalized_evidence in normalized_source:
        return True

    evidence_length = len(
        normalized_evidence
    )

    if evidence_length < 8:
        return False

    if evidence_length > len(
        normalized_source
    ):
        similarity = calculate_similarity(
            normalized_evidence,
            normalized_source,
        )

        return (
            similarity
            >= MIN_EVIDENCE_SIMILARITY
        )

    window_padding = max(
        int(
            evidence_length * 0.20
        ),
        20,
    )

    min_window_size = max(
        evidence_length - window_padding,
        1,
    )

    max_window_size = min(
        evidence_length + window_padding,
        len(normalized_source),
    )

    window_step = max(
        window_padding // 4,
        1,
    )

    position_step = max(
        evidence_length // 8,
        1,
    )

    for window_size in range(
        min_window_size,
        max_window_size + 1,
        window_step,
    ):
        last_start = (
            len(normalized_source)
            - window_size
        )

        start_positions = list(
            range(
                0,
                last_start + 1,
                position_step,
            )
        )

        # Always inspect the final possible window.
        # The normal range step may otherwise skip it.
        if (
            last_start >= 0
            and last_start
            not in start_positions
        ):
            start_positions.append(
                last_start
            )

        for start in start_positions:
            candidate = normalized_source[
                start:
                start + window_size
            ]

            similarity = calculate_similarity(
                normalized_evidence,
                candidate,
            )

            if (
                similarity
                >= MIN_EVIDENCE_SIMILARITY
            ):
                return True

    return False


def parse_entailment_decision(
    response_content: str,
) -> EntailmentDecision:
    """
    Parse the structured entailment response.
    """
    try:
        return EntailmentDecision.model_validate_json(
            response_content
        )

    except ValidationError as error:
        raise ValueError(
            "The entailment validator returned "
            "an invalid structured response."
        ) from error


def claim_is_supported_by_evidence(
    claim: str,
    evidence: str,
) -> tuple[bool, str]:
    """
    Determine whether the evidence directly supports
    the complete claim.

    This is deliberately conservative.

    Relevance is not enough.

    Example:

    Evidence:
        "Wir sollen nicht viel sprechen."

    Claim:
        "sollen expresses obligation or advice."

    This should be rejected because the example sentence
    does not explicitly establish that general rule.
    """
    system_prompt = """
You are a strict textual entailment validator.

Your task is NOT to answer the user's question.

You receive:

1. A CLAIM.
2. An EVIDENCE passage.

Determine whether the EVIDENCE directly supports the entire CLAIM.

Use ONLY the supplied evidence.

Strict rules:

1. Return supported=true only when the evidence itself is
   sufficient to justify the complete claim.

2. Do not use outside knowledge.

3. Do not infer a general rule from a single example sentence.

4. Do not assume the meaning of a word merely because the word
   occurs in the evidence.

5. If the claim contains information that is absent from the
   evidence, return supported=false.

6. If the evidence supports only part of the claim, return
   supported=false.

7. If the claim is broader or stronger than the evidence,
   return supported=false.

8. Grammatical examples do not automatically establish general
   grammatical rules.

9. Minor OCR errors may be ignored when the intended source text
   remains clear.

10. Be conservative. When uncertain, return supported=false.

Example:

CLAIM:
"Im Konjunktiv II kann sollen verwendet werden, um Ratschläge
zu geben."

EVIDENCE:
"Mit dem Modalverb sollen im Konjunktiv 2 kann man Ratschlage
geben."

Result:
supported=true

Example:

CLAIM:
"Das Verb sollen drückt eine Pflicht oder Empfehlung aus."

EVIDENCE:
"Wir sollen nicht viel sprechen. Und du?"

Result:
supported=false
""".strip()

    user_prompt = f"""
CLAIM:
{claim}

EVIDENCE:
{evidence}

Does the evidence directly support the entire claim?
""".strip()

    try:
        response = client.chat(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
            format=(
                EntailmentDecision.model_json_schema()
            ),
            options={
                "temperature": 0.0,
                "num_predict": 120,
            },
            stream=False,
        )

        decision = parse_entailment_decision(
            response.message.content
        )

        return (
            decision.supported,
            decision.reason,
        )

    except Exception as error:
        # Fail closed.
        #
        # If semantic validation itself fails,
        # the claim must not enter the final answer.
        return (
            False,
            (
                "Entailment validation failed: "
                f"{error}"
            ),
        )


def validate_claims(
    claims: list[dict],
    retrieved_chunks: list[dict],
) -> list[dict]:
    """
    Validate claims in two stages.

    Stage 1:
        Verify that the proposed evidence actually occurs
        in the referenced source.

    Stage 2:
        Verify that the evidence semantically supports
        the entire claim.

    Only claims passing BOTH stages survive.
    """
    validated_claims = []

    for claim in claims:
        source_number = claim.get(
            "source_number"
        )

        evidence = claim.get(
            "evidence",
            "",
        ).strip()

        claim_text = claim.get(
            "claim",
            "",
        ).strip()

        if not claim_text:
            continue

        if not evidence:
            continue

        if not isinstance(
            source_number,
            int,
        ):
            continue

        source_index = (
            source_number - 1
        )

        if source_index < 0:
            continue

        if source_index >= len(
            retrieved_chunks
        ):
            continue

        source_text = retrieved_chunks[
            source_index
        ].get(
            "text",
            "",
        )

        if not source_text:
            continue

        evidence_exists = (
            evidence_exists_in_source(
                evidence=evidence,
                source_text=source_text,
            )
        )

        if not evidence_exists:
            print(
                "\n[VALIDATOR] REJECTED:"
                " evidence not found"
            )

            print(
                f"Claim: {claim_text}"
            )

            print(
                f"Source: {source_number}"
            )

            print(
                f"Evidence: {evidence}"
            )

            continue

        (
            semantically_supported,
            reason,
        ) = claim_is_supported_by_evidence(
            claim=claim_text,
            evidence=evidence,
        )

        if not semantically_supported:
            print(
                "\n[VALIDATOR] REJECTED:"
                " evidence does not entail claim"
            )

            print(
                f"Claim: {claim_text}"
            )

            print(
                f"Source: {source_number}"
            )

            print(
                f"Evidence: {evidence}"
            )

            print(
                f"Reason: {reason}"
            )

            continue

        print(
            "\n[VALIDATOR] ACCEPTED"
        )

        print(
            f"Claim: {claim_text}"
        )

        print(
            f"Source: {source_number}"
        )

        print(
            f"Reason: {reason}"
        )

        validated_claims.append(
            {
                "claim": claim_text,
                "source_number": (
                    source_number
                ),
                "evidence": evidence,
            }
        )

    return validated_claims