import json
import re
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from ai_playground.llm import (
    CHAT_SYSTEM_INSTRUCTION,
    HTTP_OPTIONS,
    MODEL,
    SUMMARY_SYSTEM_INSTRUCTION,
    ask_gemini,
    ask_gemini_stream,
    create_chat_session,
    create_client,
    get_structured,
    send_chat_message,
)
from ai_playground.schemas import Summary

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


@pytest.fixture
def mock_chat() -> MagicMock:
    """A fake chat session. Tests configure send_message_stream."""
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
    """Returns the response text and sends the model, prompt, and summary instruction."""
    mock_client.models.generate_content.return_value.text = "Hello there!"

    result = ask_gemini(mock_client, "Say hello")

    assert result == "Hello there!"

    mock_client.models.generate_content.assert_called_once()
    kwargs = mock_client.models.generate_content.call_args.kwargs
    assert kwargs["model"] == MODEL
    assert kwargs["contents"] == "Say hello"
    assert kwargs["config"].system_instruction == SUMMARY_SYSTEM_INSTRUCTION


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
# Note: generators do nothing until they are looped over. Every
# streaming test consumes them with list(...), including the error
# tests; otherwise the error would never be raised.
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
    """The streaming call uses the same model, prompt, and summary instruction."""
    mock_client.models.generate_content_stream.return_value = [make_chunk("Hi")]

    list(ask_gemini_stream(mock_client, "Say hello"))

    mock_client.models.generate_content_stream.assert_called_once()
    kwargs = mock_client.models.generate_content_stream.call_args.kwargs
    assert kwargs["model"] == MODEL
    assert kwargs["contents"] == "Say hello"
    assert kwargs["config"].system_instruction == SUMMARY_SYSTEM_INSTRUCTION


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
    [[], [None], ["", None, ""]],
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


# ==========================================
# create_chat_session()
# ==========================================
def test_create_chat_session_uses_model_and_chat_instruction(mock_client):
    """The chat is created once with the chat instruction, not the summary one."""
    chat = create_chat_session(mock_client)

    assert chat is mock_client.chats.create.return_value
    mock_client.chats.create.assert_called_once()
    kwargs = mock_client.chats.create.call_args.kwargs
    assert kwargs["model"] == MODEL
    assert kwargs["config"].system_instruction == CHAT_SYSTEM_INSTRUCTION


# ==========================================
# send_chat_message()
# ==========================================
def test_send_chat_message_yields_chunks_in_order(mock_chat):
    mock_chat.send_message_stream.return_value = [
        make_chunk("Hello"),
        make_chunk(" there"),
    ]

    result = list(send_chat_message(mock_chat, "Hi"))

    assert result == ["Hello", " there"]
    mock_chat.send_message_stream.assert_called_once_with("Hi")


def test_send_chat_message_skips_empty_chunks(mock_chat):
    mock_chat.send_message_stream.return_value = [
        make_chunk("Hello"),
        make_chunk(None),
        make_chunk(""),
        make_chunk(" world"),
    ]

    result = list(send_chat_message(mock_chat, "Hi"))

    assert result == ["Hello", " world"]


@pytest.mark.parametrize(
    "chunk_texts",
    [[], [None], ["", None, ""]],
    ids=["no-chunks", "single-none", "all-empty"],
)
def test_send_chat_message_raises_when_no_text_received(mock_chat, chunk_texts):
    mock_chat.send_message_stream.return_value = [
        make_chunk(text) for text in chunk_texts
    ]

    with pytest.raises(ValueError, match=re.escape(EMPTY_RESPONSE_MESSAGE)):
        list(send_chat_message(mock_chat, "Hi"))


def test_send_chat_message_api_failure_propagates_exception(mock_chat):
    mock_chat.send_message_stream.side_effect = Exception("API Key Invalid")

    with pytest.raises(Exception, match="API Key Invalid"):
        list(send_chat_message(mock_chat, "Hi"))


# ==========================================
# get_structured()
# ==========================================
VALID_SUMMARY = {
    "title": "Why the Sky Is Blue",
    "key_points": [
        "Sunlight contains all colors of visible light.",
        "Shorter blue wavelengths scatter more in the atmosphere.",
    ],
    "sentiment": "neutral",
}


def summary_json(**overrides) -> str:
    """VALID_SUMMARY as a JSON string, with fields replaced or removed.

    Pass a field as None to remove it entirely.
    """
    data = {**VALID_SUMMARY, **overrides}
    data = {key: value for key, value in data.items() if value is not None}
    return json.dumps(data)


def test_get_structured_returns_validated_model(mock_client):
    """Valid JSON comes back as a Summary object with the right fields."""
    mock_client.models.generate_content.return_value.text = summary_json()

    result = get_structured(mock_client, "Why is the sky blue?", Summary)

    assert isinstance(result, Summary)
    assert result.title == "Why the Sky Is Blue"
    assert len(result.key_points) == 2
    assert result.sentiment == "neutral"


def test_get_structured_requests_json_with_schema(mock_client):
    """The request asks for JSON matching the schema passed in."""
    mock_client.models.generate_content.return_value.text = summary_json()

    get_structured(mock_client, "Some text", Summary)

    mock_client.models.generate_content.assert_called_once()
    kwargs = mock_client.models.generate_content.call_args.kwargs
    assert kwargs["model"] == MODEL
    assert kwargs["contents"] == "Some text"
    assert kwargs["config"].response_mime_type == "application/json"
    assert kwargs["config"].response_json_schema == Summary.model_json_schema()


@pytest.mark.parametrize(
    "bad_json",
    [
        summary_json(title=None),
        summary_json(key_points=None),
        summary_json(sentiment=None),
        summary_json(sentiment="happy"),
        summary_json(key_points=[f"Point {i}" for i in range(6)]),
        summary_json(title="x" * 101),
        "this is not json",
    ],
    ids=[
        "missing-title",
        "missing-key-points",
        "missing-sentiment",
        "invalid-sentiment",
        "too-many-key-points",
        "title-too-long",
        "not-json",
    ],
)
def test_get_structured_rejects_invalid_output(mock_client, bad_json):
    """Anything that doesn't match the schema raises ValidationError."""
    mock_client.models.generate_content.return_value.text = bad_json

    with pytest.raises(ValidationError):
        get_structured(mock_client, "Some text", Summary)


@pytest.mark.parametrize("empty_value", ["", None])
def test_get_structured_empty_response_raises_value_error(mock_client, empty_value):
    mock_client.models.generate_content.return_value.text = empty_value

    with pytest.raises(ValueError, match=re.escape(EMPTY_RESPONSE_MESSAGE)):
        get_structured(mock_client, "Some text", Summary)


def test_get_structured_api_failure_propagates_exception(mock_client):
    mock_client.models.generate_content.side_effect = Exception("API Key Invalid")

    with pytest.raises(Exception, match="API Key Invalid"):
        get_structured(mock_client, "Some text", Summary)
