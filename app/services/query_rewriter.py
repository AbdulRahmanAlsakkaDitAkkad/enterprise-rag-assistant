from ollama import Client
from pydantic import BaseModel, Field, ValidationError


OLLAMA_HOST = "http://127.0.0.1:11434"
MODEL_NAME = "qwen2.5:7b"

client = Client(host=OLLAMA_HOST)


class RewrittenQuery(BaseModel):
    query: str = Field(
        description=(
            "A standalone retrieval query that preserves the "
            "meaning of the user's current question."
        )
    )


def build_history_text(
    chat_history: list[dict] | None,
) -> str:
    """
    Convert recent conversation history into compact text.
    """
    if not chat_history:
        return "No previous conversation."

    history_parts = []

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

        history_parts.append(
            f"{label}: {content}"
        )

    if not history_parts:
        return "No previous conversation."

    return "\n".join(
        history_parts
    )


def rewrite_query(
    question: str,
    chat_history: list[dict] | None = None,
) -> str:
    """
    Rewrite a conversational follow-up into a standalone
    retrieval query.

    The rewritten query is used only for retrieval.
    The original user question is still used for final generation.
    """
    cleaned_question = question.strip()

    if not cleaned_question:
        raise ValueError(
            "The question cannot be empty."
        )

    if not chat_history:
        return cleaned_question

    history_text = build_history_text(
        chat_history
    )

    system_prompt = """
You rewrite conversational questions for document retrieval.

Your task:
Convert the current user question into a clear standalone search query.

Rules:
1. Preserve the user's meaning.
2. Use conversation history only to resolve references such as:
   "it", "they", "them", "that", "what about...", or omitted topics.
3. Do not answer the question.
4. Do not add facts.
5. Do not use outside knowledge.
6. Keep the query concise.
7. Write the rewritten query in the same language as the current question.
8. If the current question is already standalone, return it unchanged.
""".strip()

    user_prompt = f"""
Recent conversation:
{history_text}

Current question:
{cleaned_question}

Rewrite the current question as a standalone retrieval query.
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
        format=RewrittenQuery.model_json_schema(),
        options={
            "temperature": 0.0,
            "num_predict": 120,
        },
        stream=False,
    )

    try:
        rewritten = RewrittenQuery.model_validate_json(
            response.message.content
        )

    except ValidationError as error:
        raise ValueError(
            "The query rewriter returned an invalid response."
        ) from error

    rewritten_query = rewritten.query.strip()

    if not rewritten_query:
        return cleaned_question

    return rewritten_query