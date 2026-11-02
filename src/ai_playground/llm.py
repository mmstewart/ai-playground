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


def create_client() -> genai.Client:
    return genai.Client(http_options=HTTP_OPTIONS)


def ask_gemini(client: genai.Client, prompt: str) -> str:
    response = client.models.generate_content(
        model=MODEL,
        contents=prompt,
    )

    text = response.text

    if not text:
        raise ValueError(
            "Gemini returned an empty response or the content was blocked."
        )

    return text
