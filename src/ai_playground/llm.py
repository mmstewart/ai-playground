from collections.abc import Iterator

from google import genai
from google.genai import types

MODEL = "gemini-3.5-flash-lite"

HTTP_OPTIONS = types.HttpOptions(
    retry_options=types.HttpRetryOptions(
        attempts=4,
        http_status_codes=[408, 429, 500, 502, 503, 504],
    ),
    timeout=120 * 1000,
)

SYSTEM_INSTRUCTION = "You are a efficient assistant that likes to get to the point. Summarize text in 5 or less bullet points."


def create_client() -> genai.Client:
    return genai.Client(http_options=HTTP_OPTIONS)


def ask_gemini(client: genai.Client, prompt: str) -> str:
    response = client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
        ),
    )

    text = response.text

    if not text:
        raise ValueError(
            "Gemini returned an empty response or the content was blocked."
        )

    return text


def ask_gemini_stream(client: genai.Client, prompt: str) -> Iterator[str]:
    response = client.models.generate_content_stream(
        model=MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
        ),
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
