from unittest.mock import MagicMock, patch

import pytest

from ai_playground.llm import HTTP_OPTIONS, MODEL, ask_gemini, create_client


# ==========================================
# 1. Tests for create_client()
# ==========================================
def test_create_client_configures_http_options():
    """Verify that create_client initializes the SDK with the custom HTTP_OPTIONS."""
    with patch("ai_playground.llm.genai.Client") as MockClient:
        create_client()

        # Check if the Client was instantiated with the right http_options argument
        MockClient.assert_called_once_with(http_options=HTTP_OPTIONS)


# ==========================================
# 2. Tests for ask_gemini()
# ==========================================
@pytest.fixture
def mock_client():
    """Provides a mocked genai.Client instance for testing ask_gemini."""
    client = MagicMock()
    # Mock the return value of client.models.generate_content
    client.models.generate_content.return_value = MagicMock()
    return client


def test_ask_gemini_success(mock_client):
    """Verify that ask_gemini returns text correctly when given a valid response."""
    # Setup mock response text
    mock_client.models.generate_content.return_value.text = "Hello there!"

    prompt = "Say hello"
    result = ask_gemini(mock_client, prompt)

    # Assertions
    assert result == "Hello there!"
    mock_client.models.generate_content.assert_called_once_with(
        model=MODEL, contents=prompt
    )


@pytest.mark.parametrize("empty_value", ["", None])
def test_ask_gemini_empty_or_none_response_raises_value_error(mock_client, empty_value):
    """Verify that empty strings or None values trigger the ValueError exception."""
    mock_client.models.generate_content.return_value.text = empty_value

    with pytest.raises(
        ValueError,
        match="Gemini returned an empty response or the content was blocked.",
    ):
        ask_gemini(mock_client, "This will return nothing")


def test_ask_gemini_api_failure_propagates_exception(mock_client):
    """Verify that underlying API exceptions are not swallowed by ask_gemini."""
    # Simulate a network failure or bad API key error from the SDK
    mock_client.models.generate_content.side_effect = Exception("API Key Invalid")

    with pytest.raises(Exception, match="API Key Invalid"):
        ask_gemini(mock_client, "Testing error propagation")
