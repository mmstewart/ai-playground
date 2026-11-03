import re
from unittest.mock import MagicMock, patch

import pytest

from ai_playground.llm import (
    HTTP_OPTIONS,
    MODEL,
    SYSTEM_INSTRUCTION,
    ask_gemini,
    ask_gemini_stream,
    create_client,
)

EMPTY_RESPONSE_MESSAGE = "Gemini returned an empty response or the content was blocked."


def make_chunk(text: str | None) -> MagicMock:
    """Build a fake stream chunk with the given text."""
    chunk = MagicMock()
    chunk.text = text
    return chunk


@pytest.fixture
def mock_client() -> MagicMock:
    """A fake genai.Client. Tests configure the method they need."""
    return MagicMock()


# ==========================================
# create_client()
# ==========================================
def test_create_client_configures_http_options():
    """create_client builds the SDK client with our retry and timeout settings."""
    with patch("ai_playground.llm.genai.Client") as MockClient:
        create_client()

        MockClient.assert_called_once_with(http_options=HTTP_OPTIONS)


# ==========================================
# ask_gemini()
# ==========================================
def test_ask_gemini_success(mock_client):
    """Returns the response text and sends the model, prompt, and system instruction."""
    mock_client.models.generate_content.return_value.text = "Hello there!"

    result = ask_gemini(mock_client, "Say hello")

    assert result == "Hello there!"

    mock_client.models.generate_content.assert_called_once()
    kwargs = mock_client.models.generate_content.call_args.kwargs
    assert kwargs["model"] == MODEL
    assert kwargs["contents"] == "Say hello"
    assert kwargs["config"].system_instruction == SYSTEM_INSTRUCTION


@pytest.mark.parametrize("empty_value", ["", None])
def test_ask_gemini_empty_or_none_response_raises_value_error(mock_client, empty_value):
    """An empty string or None response raises ValueError."""
    mock_client.models.generate_content.return_value.text = empty_value

    with pytest.raises(ValueError, match=re.escape(EMPTY_RESPONSE_MESSAGE)):
        ask_gemini(mock_client, "This will return nothing")


def test_ask_gemini_api_failure_propagates_exception(mock_client):
    """Errors from the SDK are not swallowed."""
    mock_client.models.generate_content.side_effect = Exception("API Key Invalid")

    with pytest.raises(Exception, match="API Key Invalid"):
        ask_gemini(mock_client, "Testing error propagation")


# ==========================================
# ask_gemini_stream()
#
# Note: ask_gemini_stream is a generator, so nothing runs until it is
# looped over. Every test consumes it with list(...), including the
# error tests; otherwise the error would never be raised.
# ==========================================
def test_ask_gemini_stream_yields_chunks_in_order(mock_client):
    """Yields each chunk's text in the order it arrives."""
    mock_client.models.generate_content_stream.return_value = [
        make_chunk("Hello"),
        make_chunk(" there"),
        make_chunk("!"),
    ]

    result = list(ask_gemini_stream(mock_client, "Say hello"))

    assert result == ["Hello", " there", "!"]


def test_ask_gemini_stream_sends_model_prompt_and_system_instruction(mock_client):
    """The streaming call uses the same model, prompt, and system instruction."""
    mock_client.models.generate_content_stream.return_value = [make_chunk("Hi")]

    list(ask_gemini_stream(mock_client, "Say hello"))

    mock_client.models.generate_content_stream.assert_called_once()
    kwargs = mock_client.models.generate_content_stream.call_args.kwargs
    assert kwargs["model"] == MODEL
    assert kwargs["contents"] == "Say hello"
    assert kwargs["config"].system_instruction == SYSTEM_INSTRUCTION


def test_ask_gemini_stream_skips_empty_chunks(mock_client):
    """Chunks with no text (None or "") are skipped, not treated as errors."""
    mock_client.models.generate_content_stream.return_value = [
        make_chunk("Hello"),
        make_chunk(None),
        make_chunk(""),
        make_chunk(" world"),
    ]

    result = list(ask_gemini_stream(mock_client, "Say hello"))

    assert result == ["Hello", " world"]


@pytest.mark.parametrize(
    "chunk_texts",
    [
        [],
        [None],
        ["", None, ""],
    ],
    ids=["no-chunks", "single-none", "all-empty"],
)
def test_ask_gemini_stream_raises_when_no_text_received(mock_client, chunk_texts):
    """Raises ValueError only when the whole stream produced no text."""
    mock_client.models.generate_content_stream.return_value = [
        make_chunk(text) for text in chunk_texts
    ]

    with pytest.raises(ValueError, match=re.escape(EMPTY_RESPONSE_MESSAGE)):
        list(ask_gemini_stream(mock_client, "This will return nothing"))


def test_ask_gemini_stream_api_failure_propagates_exception(mock_client):
    """Errors from the SDK surface when the stream is consumed."""
    mock_client.models.generate_content_stream.side_effect = Exception(
        "API Key Invalid"
    )

    with pytest.raises(Exception, match="API Key Invalid"):
        list(ask_gemini_stream(mock_client, "Testing error propagation"))
