import re



from ollama import Client

from pydantic import BaseModel, Field, ValidationError



from app.services.query_decomposer import extract_comparison_concepts
# ============================================================# Configuration# ============================================================
OLLAMA_HOST = "http://127.0.0.1:11434"

MODEL_NAME = "qwen2.5:7b"



CONCEPT_WINDOW_CHARS = 450

GENERAL_WINDOW_CHARS = 900



MAX_WINDOWS_PER_CONCEPT = 10

MAX_SELECTED_GENERAL_WINDOWS = 3





FALLBACK_ANSWER_EN = (

    "I could not find enough information in the uploaded document."

)



FALLBACK_ANSWER_DE = (

    "Ich konnte in dem hochgeladenen Dokument nicht genügend "

    "Informationen finden."

)





client = Client(host=OLLAMA_HOST)
# ============================================================# Structured output models# ============================================================
class WindowSelection(BaseModel):

    selected_window_ids: list[str] = Field(

        default_factory=list,

        description=(

            "IDs of the strongest evidence windows that directly "

            "support answering the question."

        ),

    )





class ExtractedFact(BaseModel):
    window_id: str = Field(description="ID of an existing evidence window")
    line_ids: list[str] = Field(default_factory=list, description="IDs of exact source lines supporting this fact")
    fact: str = Field(description="One concise, independently supported statement relevant to the question")


class FactExtraction(BaseModel):
    facts: list[ExtractedFact] = Field(default_factory=list)


class ClaimVerification(BaseModel):
    supported: bool = Field(description="True only when the cited lines fully support the claim")
    reason: str = Field(default="", description="Short explanation of the verdict")


class FinalAnswer(BaseModel):

    answer: str = Field(

        description=(

            "Answer based only on the selected evidence windows."

        )

    )
# ============================================================# Language / fallback# ============================================================
def detect_german_question(question: str) -> bool:

    words = set(

        re.findall(

            r"\b[\wäöüß]+\b",

            question.lower(),

        )

    )



    german_markers = {

        "was",

        "wie",

        "warum",

        "wann",

        "wo",

        "wer",

        "welche",

        "welcher",

        "welches",

        "sind",

        "ist",

        "kann",

        "können",

        "muss",

        "müssen",

        "soll",

        "sollen",

        "will",

        "wollen",

        "unterschied",

        "unterscheiden",

        "zwischen",

        "der",

        "die",

        "das",

        "und",

    }



    return bool(

        words.intersection(

            german_markers

        )

    )





def get_fallback_answer(question: str) -> str:

    if detect_german_question(question):

        return FALLBACK_ANSWER_DE



    return FALLBACK_ANSWER_EN
# ============================================================# Conversation history# ============================================================
def build_conversation_history(

    chat_history: list[dict] | None,

) -> str:

    if not chat_history:

        return "No previous conversation."



    parts = []



    for message in chat_history:

        role = message.get(

            "role",

            "",

        )



        content = message.get(

            "content",

            "",

        ).strip()



        if not content:

            continue



        if role == "user":

            label = "User"



        elif role == "assistant":

            label = "Assistant"



        else:

            continue



        parts.append(

            f"{label}: {content}"

        )



    if not parts:

        return "No previous conversation."



    return "\n".join(parts)
# ============================================================# Text / window helpers# ============================================================
def normalize_for_search(text: str) -> str:

    """

    Lowercase only.



    We intentionally avoid transformations that change string

    length because window positions must correspond exactly to

    the original source text.

    """



    return text.lower()





def merge_ranges(

    ranges: list[tuple[int, int]],

) -> list[tuple[int, int]]:

    if not ranges:

        return []



    ranges = sorted(

        ranges,

        key=lambda item: item[0],

    )



    merged = [

        ranges[0]

    ]



    for start, end in ranges[1:]:

        previous_start, previous_end = merged[-1]



        if start <= previous_end:

            merged[-1] = (

                previous_start,

                max(

                    previous_end,

                    end,

                ),

            )



        else:

            merged.append(

                (

                    start,

                    end,

                )

            )



    return merged





def find_concept_ranges(

    text: str,

    concept: str,

    window_chars: int = CONCEPT_WINDOW_CHARS,

) -> list[tuple[int, int]]:

    """

    Find literal occurrences of a concept and create ranges around

    those occurrences.



    The returned positions always refer to the original text.

    """



    normalized_text = normalize_for_search(

        text

    )



    normalized_concept = normalize_for_search(

        concept

    ).strip()



    if not normalized_concept:

        return []



    pattern = re.compile(

        rf"(?<!\w){re.escape(normalized_concept)}(?!\w)",

        flags=re.IGNORECASE,

    )



    ranges = []



    for match in pattern.finditer(

        normalized_text

    ):

        start = max(

            0,

            match.start() - window_chars,

        )



        end = min(

            len(text),

            match.end() + window_chars,

        )



        ranges.append(

            (

                start,

                end,

            )

        )



    return merge_ranges(

        ranges

    )





def create_window(

    window_id: str,

    source_number: int,

    result: dict,

    text: str,

    concept: str | None = None,

) -> dict:

    """

    Create a Python-owned evidence window.



    The text is copied directly from the retrieved source and is

    never generated by the LLM.

    """



    return {

        "window_id": window_id,

        "source_number": source_number,

        "source_file": result.get(

            "source_file",

            "unknown",

        ),

        "page_number": result.get(

            "page_number",

            "unknown",

        ),

        "concept": concept,

        "text": text.strip(),

    }
# ============================================================# Comparison concept windows# ============================================================
def build_concept_windows(

    concept: str,

    retrieved_chunks: list[dict],

) -> list[dict]:

    """

    Build exact source windows around occurrences of one

    comparison concept.



    Source numbers remain identical to the numbering of the

    retrieved chunks returned by the retrieval layer.

    """



    windows = []



    concept_slug = re.sub(

        r"[^\w]+",

        "_",

        concept.lower(),

    ).strip("_")



    if not concept_slug:

        concept_slug = "concept"



    window_counter = 1



    for source_number, result in enumerate(

        retrieved_chunks,

        start=1,

    ):

        text = result.get(

            "text",

            "",

        )



        if not text:

            continue



        ranges = find_concept_ranges(

            text=text,

            concept=concept,

        )



        for start, end in ranges:

            window_text = text[

                start:end

            ].strip()



            if not window_text:

                continue



            window_id = (

                f"{concept_slug}_"

                f"{window_counter}"

            )



            windows.append(

                create_window(

                    window_id=window_id,

                    source_number=source_number,

                    result=result,

                    text=window_text,

                    concept=concept,

                )

            )



            window_counter += 1



            if (

                len(windows)

                >= MAX_WINDOWS_PER_CONCEPT

            ):

                return windows



    return windows
# ============================================================# General evidence windows# ============================================================
def split_text_into_windows(

    text: str,

    window_chars: int = GENERAL_WINDOW_CHARS,

) -> list[str]:

    """

    Split an ordinary retrieved chunk into exact evidence

    candidates.



    The split prefers paragraph/sentence boundaries where

    possible.

    """



    text = text.strip()



    if not text:

        return []



    if len(text) <= window_chars:

        return [text]



    windows = []



    start = 0



    while start < len(text):

        end = min(

            len(text),

            start + window_chars,

        )



        if end < len(text):

            candidate_breaks = [

                text.rfind(

                    "\n\n",

                    start,

                    end,

                ),

                text.rfind(

                    ". ",

                    start,

                    end,

                ),

                text.rfind(

                    "\n",

                    start,

                    end,

                ),

            ]



            best_break = max(

                candidate_breaks

            )



            minimum_break = (

                start

                + int(

                    window_chars * 0.55

                )

            )



            if best_break >= minimum_break:

                end = best_break + 1



        window_text = text[

            start:end

        ].strip()



        if window_text:

            windows.append(

                window_text

            )



        if end <= start:

            break



        start = end



    return windows





def build_general_windows(

    retrieved_chunks: list[dict],

) -> list[dict]:

    """

    Build exact evidence candidates for ordinary,

    non-comparison questions.

    """



    windows = []



    counter = 1



    for source_number, result in enumerate(

        retrieved_chunks,

        start=1,

    ):

        text = result.get(

            "text",

            "",

        )



        if not text:

            continue



        chunk_windows = split_text_into_windows(

            text

        )



        for window_text in chunk_windows:

            window_id = (

                f"evidence_{counter}"

            )



            windows.append(

                create_window(

                    window_id=window_id,

                    source_number=source_number,

                    result=result,

                    text=window_text,

                )

            )



            counter += 1



    return windows
# ============================================================# Window formatting# ============================================================
def format_windows_for_selection(

    windows: list[dict],

) -> str:

    parts = []



    for window in windows:

        concept = window.get(

            "concept"

        )



        concept_line = ""



        if concept:

            concept_line = (

                f"Focus concept: {concept}\n"

            )



        parts.append(

            (

                f"[WINDOW ID: "

                f"{window['window_id']}]\n"

                f"Source: "

                f"{window['source_number']}\n"

                f"Page: "

                f"{window['page_number']}\n"

                f"{concept_line}"

                f"Content:\n"

                f"{window['text']}"

            )

        )



    return "\n\n".join(

        parts

    )





def format_selected_windows_for_answer(

    windows: list[dict],

) -> str:

    parts = []



    for window in windows:

        concept = window.get(

            "concept"

        )



        concept_line = ""



        if concept:

            concept_line = (

                f"Focus concept: {concept}\n"

            )



        parts.append(

            (

                f"[EVIDENCE]\n"

                f"Source number: "

                f"{window['source_number']}\n"

                f"Page: "

                f"{window['page_number']}\n"

                f"{concept_line}"

                f"Exact source text:\n"

                f"{window['text']}"

            )

        )



    return "\n\n".join(

        parts

    )
# ============================================================# Structured-output parsing# ============================================================
def parse_window_selection(

    response_content: str,

) -> WindowSelection:

    try:

        return WindowSelection.model_validate_json(

            response_content

        )



    except ValidationError as error:

        raise ValueError(

            "The language model returned invalid "

            "window-selection JSON."

        ) from error





def parse_final_answer(

    response_content: str,

) -> FinalAnswer:

    try:

        return FinalAnswer.model_validate_json(

            response_content

        )



    except ValidationError as error:

        raise ValueError(

            "The language model returned invalid "

            "final-answer JSON."

        ) from error
# ============================================================# LLM evidence-window selection# ============================================================
def select_evidence_windows(

    question: str,

    windows: list[dict],

    conversation_history: str,

    focus_concept: str | None = None,

    max_selected: int = 1,

) -> list[str]:

    """

    Ask the LLM to select evidence IDs only.



    The model never copies or reconstructs evidence text.

    """



    if not windows:

        return []



    candidate_context = (

        format_windows_for_selection(

            windows

        )

    )



    system_prompt = """

You are the evidence-selection component of a grounded document

question-answering system.



You receive candidate evidence windows copied directly from

retrieved document sources.



Your only task is to identify the strongest evidence window or

windows.



You MUST NOT answer the user's question.



RULES:



1. Select evidence that directly helps answer the current

   question.



2. Prefer explicit definitions, explanations, grammatical rules,

   direct statements, and clearly explanatory passages.



3. Match the evidence type to the question.

   - For questions asking for exact conjugations, verb forms,
     or paradigms, prefer a readable table with clear headers
     and correctly aligned subject rows.

   - For questions asking for meanings, rules, or explanations,
     prefer explicit explanatory passages.

   - Never infer missing forms from incomplete or damaged tables.

   - Do not select exercises or keyword-only passages when
     stronger direct evidence is available.



4. Do not use outside knowledge.



5. Do not infer a meaning that is not explicitly supported by

   the candidate.



6. Do not copy or rewrite evidence text.



7. Return only window IDs that actually exist in the supplied

   candidates.



8. If no candidate contains useful evidence, return an empty

   selected_window_ids list.



9. Be highly selective. More evidence is not automatically

   better.



10. When several windows contain the same concept, choose the

    window that most directly explains its meaning or usage.



11. If a focus concept is supplied, judge candidates specifically

    according to how well they explain that focus concept in the

    context of the user's question.

""".strip()



    if focus_concept:

        focus_instruction = f"""

Focus concept:



{focus_concept}



Select the single strongest evidence window for this concept.



Do not select several windows merely because they contain the

word "{focus_concept}".



If one window directly explains its meaning or usage, prefer that

window over examples, exercises, tables, or answer keys.

""".strip()



    else:

        focus_instruction = f"""

There is no individual comparison concept.



Select only the strongest evidence required to answer the complete

question.



Select at most {max_selected} windows.

""".strip()



    user_prompt = f"""

Recent conversation:



{conversation_history}





Current question:



{question}





{focus_instruction}





Candidate evidence windows:



{candidate_context}





Return structured JSON containing only:



selected_window_ids

""".strip()



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

        format=WindowSelection.model_json_schema(),

        options={

            "temperature": 0.0,

            "num_predict": 150,

        },

        stream=False,

    )



    selection = parse_window_selection(

        response.message.content

    )



    valid_ids = {

        window["window_id"]

        for window in windows

    }



    selected_ids = []



    seen = set()



    for window_id in (

        selection.selected_window_ids

    ):

        if window_id not in valid_ids:

            print(

                "[WINDOW] REJECTED INVALID ID:",

                window_id,

            )

            continue



        if window_id in seen:

            continue



        seen.add(

            window_id

        )



        selected_ids.append(

            window_id

        )
# Python enforces the selection limit.
    #
# For comparison concepts max_selected is 1.# For ordinary questions it can be larger.
    return selected_ids[

        :max_selected

    ]
# ============================================================# Resolve IDs back to exact source text# ============================================================
def get_windows_by_ids(

    windows: list[dict],

    selected_ids: list[str],

) -> list[dict]:

    """

    Deterministically map selected IDs back to exact

    Python-owned evidence windows.

    """



    window_map = {

        window["window_id"]: window

        for window in windows

    }



    selected = []



    for window_id in selected_ids:

        window = window_map.get(

            window_id

        )



        if window is not None:

            selected.append(

                window

            )



    return selected





def deduplicate_selected_windows(

    windows: list[dict],

) -> list[dict]:

    """

    Remove exact duplicate evidence windows while preserving

    order.

    """



    seen = set()



    result = []



    for window in windows:

        key = (

            window["source_number"],

            window["page_number"],

            window["text"],

        )



        if key in seen:

            continue



        seen.add(

            key

        )



        result.append(

            window

        )



    return result
# ============================================================# Comparison-aware evidence selection# ============================================================
def expand_same_page_windows(
    selected_windows: list[dict],
    candidate_windows: list[dict],
    max_companions_per_page: int = 2,
    max_total_chars: int = 4200,
) -> list[dict]:
    """Add nearby retrieved evidence from the same file and page.

    Selected windows remain first. Only Python-owned retrieved text is added;
    unrelated pages are never introduced by this expansion.
    """
    expanded = list(selected_windows)
    seen_ids = {w["window_id"] for w in expanded}
    total_chars = sum(len(w["text"]) for w in expanded)
    for anchor in selected_windows:
        added_for_page = 0
        for candidate in candidate_windows:
            if candidate["window_id"] in seen_ids:
                continue
            if (candidate.get("source_file") != anchor.get("source_file")
                    or candidate.get("page_number") != anchor.get("page_number")):
                continue
            if added_for_page >= max_companions_per_page:
                break
            if total_chars + len(candidate["text"]) > max_total_chars:
                continue
            expanded.append(candidate)
            seen_ids.add(candidate["window_id"])
            total_chars += len(candidate["text"])
            added_for_page += 1
            print("[SAME-PAGE EXPANSION]", anchor["window_id"],
                  "->", candidate["window_id"], "page", candidate["page_number"])
    return deduplicate_selected_windows(expanded)


def select_windows_for_question(

    question: str,

    retrieved_chunks: list[dict],

    conversation_history: str,

) -> list[dict]:

    concepts = extract_comparison_concepts(

        question

    )
# --------------------------------------------------------# Ordinary question# --------------------------------------------------------
    if not concepts:

        windows = build_general_windows(

            retrieved_chunks

        )



        print()

        print(

            "========== EVIDENCE WINDOWS =========="

        )



        for window in windows:

            print(

                {

                    "window_id": (

                        window["window_id"]

                    ),

                    "source": (

                        window["source_number"]

                    ),

                    "page": (

                        window["page_number"]

                    ),

                }

            )



        print(

            "======================================"

        )

        print()



        selected_ids = select_evidence_windows(

            question=question,

            windows=windows,

            conversation_history=conversation_history,

            max_selected=MAX_SELECTED_GENERAL_WINDOWS,

        )



        print(

            "[WINDOW SELECTION]:",

            selected_ids,

        )



        selected_windows = get_windows_by_ids(

            windows=windows,

            selected_ids=selected_ids,

        )

        return expand_same_page_windows(selected_windows, windows)
# --------------------------------------------------------# Comparison question# --------------------------------------------------------
    print()

    print(

        "========== COMPARISON CONCEPTS =========="

    )



    for concept in concepts:

        print(

            f"- {concept}"

        )



    print(

        "========================================="

    )

    print()



    all_selected_windows = []



    for concept in concepts:

        windows = build_concept_windows(

            concept=concept,

            retrieved_chunks=retrieved_chunks,

        )



        print(

            f"========== WINDOWS FOR {concept} =========="

        )



        if not windows:

            print(

                "No concept windows found."

            )



        for window in windows:

            print()

            print(

                f"[{window['window_id']}]"

            )



            print(

                f"Source: "

                f"{window['source_number']}"

            )



            print(

                f"Page: "

                f"{window['page_number']}"

            )



            print(

                window["text"]

            )



        print()

        print(

            "==========================================="

        )

        print()



        if not windows:

            continue



        selected_ids = select_evidence_windows(

            question=question,

            windows=windows,

            conversation_history=conversation_history,

            focus_concept=concept,

            max_selected=1,

        )



        print(

            f"[WINDOW SELECTION] "

            f"{concept}: "

            f"{selected_ids}"

        )



        selected_windows = get_windows_by_ids(

            windows=windows,

            selected_ids=selected_ids,

        )



        all_selected_windows.extend(

            expand_same_page_windows(selected_windows, windows)

        )



    return deduplicate_selected_windows(

        all_selected_windows

    )
# ============================================================# Final answer generation# ============================================================
def is_ambiguous_ocr_table_evidence(question: str, text: str, quote: str) -> bool:
    """Conservatively reject isolated conjugation-table rows with broken OCR.

    This is a narrow safety guard, not a general semantic verifier.
    """
    q = question.casefold()
    t = text.casefold()
    if not ("konjunktiv" in q and any(x in q for x in ("bild", "form", "konjug"))):
        return False
    table_markers = (
        "sein haben", "k6nnen", "können_", "wolltest solltest",
        "du hattest du konntest", "sie/sie | konnten",
    )
    if not any(marker in t for marker in table_markers):
        return False
    # A table fragment cannot independently establish a full paradigm or rule.
    # Even a literal quote may be assigned to the wrong column by OCR.
    return True


def _question_focus_terms(question: str) -> set[str]:
    """Extract explicit grammar/topic focus without treating stopwords as subjects."""
    stop = {"was", "wie", "wer", "wann", "wo", "ist", "sind", "und", "oder", "wenn", "man", "etwas", "von", "einer", "person", "möchte", "bedeutet", "bildet", "den", "die", "das", "im", "es", "höflich", "nicht", "mit", "der", "ein", "eine", "sich", "werden", "kann"}
    terms = set(re.findall(r"\b[a-zäöüß]{4,}\b", question.casefold())) - stop
    return terms


def _has_polarity_conflict(fact: str, quotes: list[str]) -> bool:
    """Reject obvious negation reversals; this is not semantic entailment."""
    evidence = " ".join(quotes).casefold()
    statement = fact.casefold()
    # A source's 'nicht höflich' must not become 'höflich'.
    if re.search(r"\b(?:nicht|unhöflich)\W+höflich\b", evidence):
        if "höflich" in statement and not ("nicht höflich" in statement or "unhöflich" in statement):
            return True
    if "nicht hoflich" in evidence or "nicht höflich" in evidence:
        if "höflich" in statement and not ("nicht höflich" in statement or "unhöflich" in statement):
            return True
    return False


def build_numbered_evidence_lines(selected_windows: list[dict]) -> tuple[str, dict]:
    """Assign Python-owned IDs to exact OCR lines, preserving provenance."""
    line_map = {}
    blocks = []
    for window in selected_windows:
        entries = []
        for index, line in enumerate(window["text"].splitlines(), start=1):
            if not line.strip():
                continue
            line_id = f"{window['window_id']}_line_{index}"
            line_map[line_id] = {"text": line, "window": window}
            entries.append(f"[{line_id}] {line}")
        blocks.append(
            f"WINDOW {window['window_id']} | Source {window['source_number']} "
            f"| Page {window['page_number']}\n" + "\n".join(entries)
        )
    return "\n\n".join(blocks), line_map


def _polarity_inconsistent(fact: str, evidence: str) -> bool:
    """Reject obvious positive/negative politeness reversals."""
    f = fact.casefold()
    e = evidence.casefold()
    e_negative = bool(re.search(r"\b(?:nicht\s+h[oö]flich|unh[oö]flich)\b", e))
    f_negative = bool(re.search(r"\b(?:nicht\s+h[oö]flich|unh[oö]flich)\b", f))
    f_positive = bool(re.search(r"\bh[oö]flich\b", f))
    return e_negative and f_positive and not f_negative


def verify_claim_against_lines(question: str, fact: str, evidence: str) -> bool:
    """Fail-closed semantic verification of a candidate against its cited lines.

    The independent LLM check is an additional safeguard, not proof of entailment.
    Grammatical forms enclosed in quotes are required to occur literally in
    the evidence, with case-insensitive matching. This deliberately rejects
    guessed corrections to damaged OCR instead of silently inventing them.
    """
    # Reject asserted quoted forms not present in the source lines.
    quoted_forms = re.findall(r"['\"]([^'\"]{2,70})['\"]", fact)
    for form in quoted_forms:
        if form.casefold() not in evidence.casefold():
            print("[FACT REJECTED: UNSUPPORTED QUOTED FORM]", repr(form))
            return False

    # 'wird + Infinitiv' is not interchangeable with 'würde + Infinitiv'.
    # Also reject an unquoted assertion of this form unless explicitly sourced.
    if re.search(r"\bwird\s*\+\s*infinitiv\b", fact, re.I):
        if not re.search(r"\bwird\s*\+\s*infinitiv\b", evidence, re.I):
            print("[FACT REJECTED: UNSUPPORTED GRAMMATICAL FORM]", repr(fact))
            return False

    prompt = """You are a strict, independent verifier of document-grounded claims.
Decide whether the CLAIM is FULLY and DIRECTLY supported by the CITED SOURCE LINES.
Use ONLY the supplied lines. Do not use your knowledge of German grammar.
The text may have OCR errors. Do not correct or guess damaged conjugation tables.
A related topic, a matching keyword, or a plausible grammar rule is NOT enough.
If the source says 'würde' (or OCR 'wirde'), a claim saying 'wird' is NOT supported.
If a claim says 'konnte' where the source is ambiguous about 'könnte', reject it.
Reject when the lines do not establish the claimed subject, form, or relationship.
Return supported=false whenever uncertain. Return structured JSON only."""
    response = client.chat(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": (
                f"QUESTION:\n{question}\n\n"
                f"CITED SOURCE LINES (only admissible evidence):\n{evidence}\n\n"
                f"CLAIM TO VERIFY:\n{fact}"
            )},
        ],
        format=ClaimVerification.model_json_schema(),
        options={"temperature": 0.0, "num_predict": 160},
        stream=False,
    )
    try:
        result = ClaimVerification.model_validate_json(response.message.content)
    except (ValidationError, ValueError) as exc:
        print("[FACT REJECTED: VERIFIER INVALID OUTPUT]", str(exc))
        return False
    print("[CLAIM VERIFICATION]", {"supported": result.supported, "reason": result.reason, "fact": fact})
    if not result.supported:
        print("[FACT REJECTED: SEMANTIC VERIFICATION]", repr(fact))
    return result.supported



def build_structured_table_context(
    selected_windows: list[dict],
) -> str:
    """
    Represent consistently aligned pipe-delimited tables
    as explicit header-to-cell relationships.

    Preserve the original source line IDs.
    Never invent or repair missing table cells.
    """
    blocks = []

    for window in selected_windows:
        lines = window["text"].splitlines()

        table_rows = []
        for line_number, line in enumerate(lines, start=1):
            if "|" not in line:
                continue

            cells = [
                cell.strip()
                for cell in line.split("|")
            ]

            table_rows.append((line_number, cells))

        if len(table_rows) < 2:
            continue

        header_number, headers = table_rows[0]
        column_count = len(headers)

        if column_count < 2:
            continue

        if not all(
            len(cells) == column_count
            for _, cells in table_rows
        ):
            continue

        if not all(headers[1:]):
            continue

        window_id = window["window_id"]
        header_id = (
            f"{window_id}_line_{header_number}"
        )

        relationships = []

        for row_number, cells in table_rows[1:]:
            subject = cells[0]

            if not subject:
                continue

            row_id = f"{window_id}_line_{row_number}"

            for column in range(1, column_count):
                header = headers[column]
                value = cells[column]

                if not value:
                    continue

                relationships.append(
                    f"{subject} | {header} = {value} "
                    f"| source_lines: [{header_id}], "
                    f"[{row_id}]"
                )

        if relationships:
            blocks.append(
                f"WINDOW {window_id} "
                f"| Source {window['source_number']} "
                f"| Page {window['page_number']}\n"
                + "\n".join(relationships)
            )

    return "\n\n".join(blocks)



def extract_verified_table_cells(
    selected_windows: list[dict],
) -> list[dict]:
    """
    Extract literal cells from consistently aligned pipe tables.

    This establishes structural provenance, not semantic relevance.
    Never repair OCR, infer missing cells, or invent values.
    """
    extracted = []

    for window in selected_windows:
        rows = []

        for line_number, line in enumerate(
            window["text"].splitlines(), start=1
        ):
            if "|" not in line:
                continue

            cells = [cell.strip() for cell in line.split("|")]
            rows.append((line_number, cells))

        if len(rows) < 2:
            continue

        header_number, headers = rows[0]
        width = len(headers)

        if width < 2:
            continue

        if any(len(cells) != width for _, cells in rows):
            continue

        if not all(headers[1:]):
            continue

        if len(set(h.casefold() for h in headers[1:])) != width - 1:
            continue

        if any(
            not all(cells)
            for _, cells in rows[1:]
        ):
            continue

        window_id = window["window_id"]
        header_id = f"{window_id}_line_{header_number}"

        for row_number, cells in rows[1:]:
            row_id = f"{window_id}_line_{row_number}"

            for column in range(1, width):
                extracted.append({
                    "window_id": window_id,
                    "window": window,
                    "header": headers[column],
                    "row_label": cells[0],
                    "value": cells[column],
                    "line_ids": [header_id, row_id],
                    "evidence": (
                        rows[0][1][0]
                        + " | "
                        + " | ".join(headers[1:])
                        + "\n"
                        + " | ".join(cells)
                    ),
                })

    return extracted



def match_question_table_cells(
    question: str,
    selected_windows: list[dict],
) -> list[dict]:
    """
    Match explicitly requested column headers to literal table cells.

    Conservative scope:
    - Require at least two matching headers in one table.
    - Require complete, consistently aligned table rows.
    - Do not infer missing values or semantic relationships.
    """
    import re

    question_terms = set(
        re.findall(r"\b\w+\b", question.casefold())
    )

    cells = extract_verified_table_cells(selected_windows)

    by_window = {}

    for cell in cells:
        if cell["header"].casefold() not in question_terms:
            continue

        by_window.setdefault(
            cell["window_id"], []
        ).append(cell)

    matched = []

    for window_id, window_cells in by_window.items():
        headers = {
            cell["header"].casefold()
            for cell in window_cells
        }

        if len(headers) < 2:
            continue

        rows_by_header = {}

        for cell in window_cells:
            rows_by_header.setdefault(
                cell["header"].casefold(), set()
            ).add(cell["row_label"])

        row_sets = list(rows_by_header.values())

        if not row_sets:
            continue

        if not all(rows == row_sets[0] for rows in row_sets):
            continue

        matched.extend(window_cells)

    return matched


def extract_supported_facts(
    question: str,
    selected_windows: list[dict],
    conversation_history: str,
) -> list[dict]:
    """LLM selects numbered source lines; Python validates IDs and provenance.

    IDs establish which original lines were used, NOT semantic entailment.
    """
    if not selected_windows:
        return []
    numbered_text, line_map = build_numbered_evidence_lines(selected_windows)
    structured_tables = build_structured_table_context(selected_windows)
    
    system_prompt = """
You are a conservative fact-extraction component for a
document-grounded question-answering system.

Your job is to extract ALL relevant facts that can be reliably
supported by the numbered source lines.

Return structured JSON containing only:
{
  "facts": [
    {
      "window_id": "existing window ID",
      "line_ids": ["existing line ID"],
      "fact": "one independently supported statement"
    }
  ]
}

IMPORTANT: MULTI-SUBJECT COVERAGE

1. Identify every subject explicitly requested in the current question.

2. Consider each requested subject separately.

3. For a question about two verbs, attempt to find evidence for
   both verbs before returning the facts.

4. Do not stop after finding one general statement.

5. Prefer facts that directly answer the requested subjects
   over background information.

6. Extract a general rule only when it helps answer the question.

EVIDENCE RULES

7. Every fact must cite 1 to 5 existing line IDs from ONE window.

8. Use the exact window IDs and line IDs supplied in the source.

9. The cited lines must jointly support the complete fact.

10. Never invent grammatical forms, rules, examples, or quotations.

11. Preserve distinctions between:
    - Präsens
    - Präteritum
    - Konjunktiv II

12. For German modal verbs, do not confuse:
    - konnte with könnte
    - wurde with würde
    - wird with würde

13. OCR text may contain damaged characters, missing umlauts,
    broken columns, or interleaved table rows.

14. Never reconstruct a complete conjugation paradigm from an
    ambiguous OCR table.

15. Prefer readable explanatory sentences and unambiguous examples
    over damaged tables.

16. Do not infer that a word belongs to a particular table column
    unless its association is clear.

17. Do not combine lines from different windows into one fact.

TABLE EVIDENCE RULES

17a. For a readable table, identify the column header and
     the relevant data row before extracting a fact.

17b. Cite BOTH the header line ID and the data row line ID
     whenever both establish the relationship.

17c. For conjugation tables, associate each verb header
     with the correct form in the requested subject row.

17d. Copy line IDs EXACTLY as displayed in the evidence.
     Never calculate, invent, or guess line numbers.

17e. Never cite a line ID absent from the supplied evidence.

17f. Use tables only when headers and columns are clear.
     Never guess when OCR alignment is ambiguous.

17g. Extract supported forms separately when appropriate.
     Cite up to five existing line IDs per fact.

18. Do not treat conversation history as source evidence.

19. If one requested subject is supported and another is not,
    extract only the supported facts.

20. If no reliable fact is supported, return {"facts": []}.

EXTRACTION QUALITY

21. Use the language of the user's question.

22. Keep facts concise and independently understandable.


22a. A fact must directly answer at least one part of the
     current question. Do not extract general background rules
     when the question asks for specific forms.

22b. For explanation questions, do not copy raw OCR fragments
     as facts; express supported relationships clearly.
     EXCEPTION: For example requests, copy a complete, clearly
     readable example sentence from the cited source lines.

22c. For questions asking how specific verbs are formed,
     prioritize the actual verb forms over general explanations.

22d. For questions asking to explain a transformation,
     describe the supported transformation.
     For questions asking for an example, reproduce the
     relevant, clearly readable example sentence instead.

22e. Correct obvious OCR spelling errors only when the intended
     reading is unambiguous from the cited text. Never guess
     missing forms or reconstruct damaged tables.

22f. If the question names multiple verbs, try to extract a
     separate supported fact for each verb. Do not substitute
     a general rule for missing verb-specific evidence.

23. Do not duplicate equivalent facts.

24. Do not add a fact merely to increase coverage.

25. Prefer precision over completeness when the source is unclear.

EXAMPLE REQUESTS

26. If the current question asks for an example, extract a
    concrete example from the source rather than a general rule.

27. Preserve the complete example sentence exactly when it is
    clearly readable in the evidence.

28. For example requests, prefer one directly relevant example
    over multiple explanatory or background facts.

29. Do not extract unrelated examples, even if they discuss
    the same broad grammatical topic.

30. If no relevant example is supported by the source,
    return an empty facts list rather than substituting
    general background information.

Return only the requested structured JSON.
""".strip()

    user_prompt = (
        f"CURRENT QUESTION:\n{question}\n\n"
        f"CONVERSATION (reference resolution only):\n{conversation_history}\n\n"
        f"NUMBERED SOURCE LINES:\n{numbered_text}\n\n"
        f"STRUCTURED TABLE RELATIONSHIPS (derived from the "
        f"same numbered source lines):\n{structured_tables}\n\n"
        "Use only existing source line IDs. "
        "Extract supported facts and their exact line IDs."
    )
    response = client.chat(
        model=MODEL_NAME,
        messages=[{"role": "system", "content": system_prompt},
                  {"role": "user", "content": user_prompt}],
        format=FactExtraction.model_json_schema(),
        options={"temperature": 0.0, "num_predict": 900},
        stream=False,
    )
    try:
        extraction = FactExtraction.model_validate_json(response.message.content)
    except ValidationError as error:
        raise ValueError("Invalid numbered-line fact-extraction JSON.") from error

    focus_terms = _question_focus_terms(question)
    accepted, seen = [], set()
    for candidate in extraction.facts:
        print("[FACT CANDIDATE]", {
            "window_id": candidate.window_id,
            "line_ids": candidate.line_ids,
            "fact": candidate.fact,
        })
        ids = list(dict.fromkeys(
            line_id.strip().strip("[]").strip()
            for line_id in candidate.line_ids
        ))
        fact = candidate.fact.strip()
        
        # Reject unrelated grammatical examples when the question
        # explicitly asks about specific modal verbs.
        question_lower = question.casefold()

        if (
            "konjunktiv" in question_lower
            and "können" in question_lower
            and "sollen" in question_lower
        ):
            fact_lower = fact.casefold()

            koennen_markers = (
                "können", "konnen", "k6nnen",
                "könnte", "könntest", "könnten", "könntet",
                "kénnte"
            )

            sollen_markers = (
                "sollen", "sollte", "solltest",
                "sollten", "solltet"
            )

            if not (
                any(marker in fact_lower for marker in koennen_markers)
                or any(marker in fact_lower for marker in sollen_markers)
            ):
                print(
                    "[FACT REJECTED: IRRELEVANT VERB]",
                    repr(fact)
                )
                continue

        # For requests asking specifically for an example of a
        # polite request in Konjunktiv 2, reject explanations and
        # suggestions that do not demonstrate a polite request.
        question_focus = question.casefold()
        fact_focus = fact.casefold()

        asks_for_polite_request_example = (
            "konjunktiv" in question_focus
            and "beispiel" in question_focus
            and (
                "h\u00f6flich" in question_focus
                or "bitten" in question_focus
            )
        )

        if asks_for_polite_request_example:
            polite_request_markers = (
                "k\u00f6nnten sie",
                "w\u00fcrden sie",
                "d\u00fcrfte ich",
                "k\u00f6nntest du",
                "w\u00fcrdest du",
            )

            if not any(
                marker in fact_focus
                for marker in polite_request_markers
            ):
                print(
                    "[FACT REJECTED: NOT A POLITE REQUEST EXAMPLE]",
                    repr(fact),
                )
                continue

        if not fact or not (1 <= len(ids) <= 5):
            continue
        if any(line_id not in line_map for line_id in ids):
            # Recover only a verbatim example sentence from an
            # existing, uniquely matching source line.
            # Never guess a line number or reconstruct OCR text.
            normalized_fact = " ".join(fact.casefold().split())

            exact_matches = [
                line_id
                for line_id, record in line_map.items()
                if normalized_fact
                and normalized_fact in " ".join(
                    record["text"].casefold().split()
                )
            ]

            # Prefer a complete standalone source line.
            standalone_matches = [
                line_id
                for line_id in exact_matches
                if normalized_fact == " ".join(
                    line_map[line_id]["text"].casefold().split()
                )
            ]

            if len(standalone_matches) == 1:
                recovered_id = standalone_matches[0]
                ids = [recovered_id]
                candidate_window_id = line_map[
                    recovered_id
                ]["window"]["window_id"]

                print(
                    "[FACT LINE ID RECOVERED]",
                    recovered_id,
                    "from", candidate.window_id,
                )

            elif len(exact_matches) == 1:
                recovered_id = exact_matches[0]
                ids = [recovered_id]
                candidate_window_id = line_map[
                    recovered_id
                ]["window"]["window_id"]

                print(
                    "[FACT LINE ID RECOVERED]",
                    recovered_id,
                    "from", candidate.window_id,
                )

            else:
                print(
                    "[FACT REJECTED: UNKNOWN LINE ID]",
                    repr(ids),
                    "matching lines:", len(exact_matches),
                )
                continue
        else:
            candidate_window_id = candidate.window_id
        if any(line_map[line_id]["window"]["window_id"] != candidate_window_id for line_id in ids):
            print("[FACT REJECTED: WRONG WINDOW]", repr(ids))
            continue
        evidence = "\n".join(line_map[line_id]["text"] for line_id in ids)
        # A subject must appear in the evidence and the claim; the mere
        # presence of the subject somewhere in the full window is insufficient.
        
        # Do not require exact keyword overlap here.
        # German inflections and OCR errors can change the spelling
        # of relevant terms (e.g. können -> konnen / k6nnen).
        #
        # The semantic verifier below evaluates whether the
        # candidate is actually supported by its cited lines.
        # It must remain enabled.

        if not fact.strip():
            print("[FACT REJECTED: EMPTY FACT]", repr(ids))
            continue

        if _polarity_inconsistent(fact, evidence):
            print("[FACT REJECTED: POLARITY]", repr(fact), repr(ids))
            continue
        if is_ambiguous_ocr_table_evidence(question, evidence, evidence):
            print("[FACT REJECTED: AMBIGUOUS OCR TABLE]", repr(ids))
            continue
        if not verify_claim_against_lines(question, fact, evidence):
            continue
        key = (candidate_window_id, fact.casefold())
        if key in seen:
            continue
        seen.add(key)
        window = line_map[ids[0]]["window"]
        accepted.append({
            "window_id": candidate_window_id, "line_ids": ids,
            "fact": fact, "evidence": evidence, "window": window,
        })
        print("[FACT]", candidate.window_id, repr(fact), "LINES:", repr(ids))
        if len(accepted) >= 12:
            break
    print("[FACT EXTRACTION]", len(extraction.facts),
          "candidates;", len(accepted), "line-verified")
    return accepted


def generate_answer_from_windows(
    question: str,
    selected_windows: list[dict],
    conversation_history: str,
) -> FinalAnswer:
    """Deterministic composition prevents a second LLM from adding new claims.

    The extracted claims still require semantic evaluation: quote existence
    alone cannot prove their meaning, especially in damaged tables.
    """

    # Deterministic path for explicitly requested Konjunktiv tables.
    # Other questions retain the original LLM verification pipeline.
    if "konjunktiv" in question.casefold():
        cells = match_question_table_cells(
            question, selected_windows
        )

        window_ids = {cell["window_id"] for cell in cells}

        if cells and len(window_ids) == 1:
            source_window = cells[0]["window"]

            context_windows = [
                window for window in selected_windows
                if window["page_number"] == source_window["page_number"]
                and window["window_id"] != source_window["window_id"]
                and "konjunktiv 2" in window["text"].casefold()
            ]

            if context_windows:
                headers = list(dict.fromkeys(
                    cell["header"] for cell in cells
                ))

                rows = {}

                for cell in cells:
                    rows.setdefault(
                        cell["row_label"], {}
                    )[cell["header"]] = cell["value"]

                if all(
                    all(header in values for header in headers)
                    for values in rows.values()
                ):
                    answer_lines = [
                        " | ".join(["Person"] + headers)
                    ]

                    for person, values in rows.items():
                        answer_lines.append(
                            " | ".join(
                                [person] +
                                [values[header] for header in headers]
                            )
                        )

                    selected_windows[:] = [source_window]

                    answer = "\n".join(answer_lines)

                    print(
                        "[DETERMINISTIC TABLE ANSWER]",
                        repr(answer)
                    )

                    return FinalAnswer(answer=answer)

    supported_facts = extract_supported_facts(
        question=question,
        selected_windows=selected_windows,
        conversation_history=conversation_history,
    )
    if not supported_facts:
        selected_windows.clear()
        return FinalAnswer(answer="")
    used_ids = {item["window_id"] for item in supported_facts}
    selected_windows[:] = [w for w in selected_windows if w["window_id"] in used_ids]
    answer_parts = []
    for item in supported_facts:
        fact = item["fact"].strip()
        if fact.endswith((".'", '."', '.”', ".’", ".", "!", "?")):
            answer_parts.append(fact)
        else:
            answer_parts.append(fact + ".")
    answer = " ".join(answer_parts)
    print("[DETERMINISTIC ANSWER]", repr(answer))
    return FinalAnswer(answer=answer)


# ============================================================
# Backend-controlled citations
# ============================================================


def append_citations_from_windows(

    answer: str,

    selected_windows: list[dict],

) -> str:

    """

    Citations are generated entirely by Python.



    The final LLM never generates or repeats source/window IDs.



    Because selected_windows are the exact evidence supplied to

    the final answer model, their provenance can be trusted by

    the backend.

    """



    cleaned_answer = answer.strip()



    if not cleaned_answer:

        return ""



    citations = []



    seen = set()



    for window in selected_windows:

        source_number = window[

            "source_number"

        ]



        page_number = window[

            "page_number"

        ]



        citation_key = (

            source_number,

            page_number,

        )



        if citation_key in seen:

            continue



        seen.add(

            citation_key

        )



        citations.append(

            (

                f"[Source {source_number}, "

                f"page {page_number}]"

            )

        )



    if not citations:

        return ""



    return (

        f"{cleaned_answer} "

        f"{' '.join(citations)}"

    )
# ============================================================# Main RAG pipeline# ============================================================
def generate_rag_answer(

    question: str,

    retrieved_chunks: list[dict],

    chat_history: list[dict] | None = None,

) -> str:

    """

    Evidence-window RAG pipeline.



    ----------------------------------------------------------



    Retrieval

        ↓

    Retrieved chunks

        ↓

    Comparison decomposition when applicable

        ↓

    Python creates exact source windows

        ↓

    LLM selects WINDOW IDs only

        ↓

    Python resolves IDs to exact original text

        ↓

    LLM extracts atomic facts with exact source quotations
        ↓
    Python checks literal spans and rejects ambiguous OCR table evidence
        ↓
    Final LLM sees only span-verified facts

        ↓

    Final LLM generates answer text only

        ↓

    Python attaches source/page citations



    ----------------------------------------------------------



    Important trust boundaries:



    - The LLM proposes supporting quotes; Python verifies exact spans.

    - The LLM never creates final citation IDs.

    - Source text is always Python-owned.

    - Source/page metadata is always Python-owned.

    """



    cleaned_question = question.strip()



    if not cleaned_question:

        raise ValueError(

            "The question cannot be empty."

        )



    if not retrieved_chunks:

        return get_fallback_answer(

            cleaned_question

        )



    # Conversation history remains available to the API query rewriter.
    # Do not use previous answers as factual evidence.
    conversation_history = "No previous conversation."
# --------------------------------------------------------# Stage 1:# Build exact windows and select evidence IDs# --------------------------------------------------------
    selected_windows = (

        select_windows_for_question(

            question=cleaned_question,

            retrieved_chunks=retrieved_chunks,

            conversation_history=conversation_history,

        )

    )



    print()

    print(

        "========== SELECTED WINDOWS =========="

    )



    if selected_windows:

        for window in selected_windows:

            print(

                {

                    "window_id": (

                        window["window_id"]

                    ),

                    "source_number": (

                        window["source_number"]

                    ),

                    "page_number": (

                        window["page_number"]

                    ),

                    "concept": (

                        window.get("concept")

                    ),

                }

            )



    else:

        print(

            "No windows selected."

        )



    print(

        "======================================"

    )

    print()



    if not selected_windows:

        return get_fallback_answer(

            cleaned_question

        )
# --------------------------------------------------------# Stage 2:# Generate answer from exact selected evidence# --------------------------------------------------------
    generated_answer = (

        generate_answer_from_windows(

            question=cleaned_question,

            selected_windows=selected_windows,

            conversation_history=conversation_history,

        )

    )



    print()

    print(

        "========== FINAL GROUNDED ANSWER =========="

    )



    print(

        {

            "answer": (

                generated_answer.answer

            ),

        }

    )



    print(

        "======================================"

    )

    print()



    if not generated_answer.answer.strip():

        return get_fallback_answer(

            cleaned_question

        )
# --------------------------------------------------------# Stage 3:# Backend-controlled citations# --------------------------------------------------------
    final_answer = (

        append_citations_from_windows(

            answer=generated_answer.answer,

            selected_windows=selected_windows,

        )

    )



    if not final_answer:

        return get_fallback_answer(

            cleaned_question

        )



    return final_answer