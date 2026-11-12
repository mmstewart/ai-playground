from collections.abc import Iterator

from google import genai
from google.genai import types
from google.genai.chats import Chat
from pydantic import BaseModel

MODEL = "gemini-3.5-flash-lite"

HTTP_OPTIONS = types.HttpOptions(
    retry_options=types.HttpRetryOptions(
        attempts=4,
        http_status_codes=[408, 429, 500, 502, 503, 504],
    ),
    timeout=120 * 1000,
)

SUMMARY_SYSTEM_INSTRUCTION = (
    "You are an efficient assistant that gets to the point. "
    "Summarize the text in five or fewer bullet points."
)

CHAT_SYSTEM_INSTRUCTION = (
    "You are a friendly, knowledgeable assistant in a back-and-forth conversation. "
    "Answer clearly and concisely, and use earlier messages for context."
)

SUMMARY_CONFIG = types.GenerateContentConfig(
    system_instruction=SUMMARY_SYSTEM_INSTRUCTION,
)

CHAT_CONFIG = types.GenerateContentConfig(
    system_instruction=CHAT_SYSTEM_INSTRUCTION,
)


def create_client() -> genai.Client:
    return genai.Client(http_options=HTTP_OPTIONS)


def create_chat_session(client: genai.Client) -> Chat:
    return client.chats.create(
        model=MODEL,
        config=CHAT_CONFIG,
    )


def ask_gemini(client: genai.Client, prompt: str) -> str:
    if not prompt.strip():
        raise ValueError("Prompt is empty.")

    response = client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config=SUMMARY_CONFIG,
    )

    text = response.text

    if not text:
        raise ValueError(
            "Gemini returned an empty response or the content was blocked."
        )

    return text


async def ask_gemini_async(client: genai.Client, prompt: str) -> str:
    if not prompt.strip():
        raise ValueError("Prompt is empty.")

    response = await client.aio.models.generate_content(
        model=MODEL,
        contents=prompt,
        config=SUMMARY_CONFIG,
    )

    text = response.text

    if not text:
        raise ValueError(
            "Gemini returned an empty response or the content was blocked."
        )

    return text


def ask_gemini_stream(client: genai.Client, prompt: str) -> Iterator[str]:
    if not prompt.strip():
        raise ValueError("Prompt is empty.")

    response = client.models.generate_content_stream(
        model=MODEL,
        contents=prompt,
        config=SUMMARY_CONFIG,
    )

    received_text = False

    for chunk in response:
        if chunk.text:
            received_text = True
            yield chunk.text

    if not received_text:
        raise ValueError(
            "Gemini returned an empty response or the content was blocked."
        )


def get_structured[T: BaseModel](
    client: genai.Client, prompt: str, schema: type[T]
) -> T:
    if not prompt.strip():
        raise ValueError("Prompt is empty.")

    response = client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_json_schema=schema.model_json_schema(),
        ),
    )

    text = response.text

    if not text:
        raise ValueError(
            "Gemini returned an empty response or the content was blocked."
        )

    return schema.model_validate_json(text)


def send_chat_message(chat: Chat, prompt: str) -> Iterator[str]:
    response = chat.send_message_stream(prompt)

    received_text = False

    for chunk in response:
        if chunk.text:
            received_text = True
            yield chunk.text

    if not received_text:
        raise ValueError(
            "Gemini returned an empty response or the content was blocked."
        )
