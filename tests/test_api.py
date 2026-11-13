import json
from collections.abc import Iterator
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient
from google.genai import errors

from ai_playground import api
from ai_playground.api import app, get_client
from ai_playground.llm import EmptyPromptError
from ai_playground.schemas import MeetingNotes, Summary

VALID_TEXT = "The team met to plan the next release and agreed on the scope."

SUMMARY_JSON = json.dumps(
    {
        "title": "Release Planning",
        "key_points": ["The team agreed on the release scope."],
        "sentiment": "positive",
    }
)

MEETING_NOTES_JSON = json.dumps(
    {
        "title": "Fitness App Sprint Meeting",
        "decisions": ["Approved the dashboard redesign."],
        "risks": [
            {"department": "Engineering", "risks": ["Wearable API rate limits."]}
        ],
        "action_items": [{"task": "Finalize wireframes.", "owner": "Sarah"}],
    }
)


# ==========================================
# Fixtures
#
# TestClient(app) is used without a `with` block on purpose: the
# lifespan (which creates the real Gemini client) only runs inside
# `with TestClient(app)`. The real client is never created, so these
# tests need no API key, and the mock is injected through
# dependency_overrides instead.
# ==========================================
@pytest.fixture
def mock_client() -> Iterator[MagicMock]:
    """A fake Gemini client injected into every route through get_client."""
    client = MagicMock()
    client.aio.models.generate_content = AsyncMock()

    app.dependency_overrides[get_client] = lambda: client
    yield client
    app.dependency_overrides.clear()


@pytest.fixture
def http() -> TestClient:
    return TestClient(app)


def gemini_returns(mock_client: MagicMock, text: str | None) -> None:
    """Make the fake Gemini call return the given response text."""
    mock_client.aio.models.generate_content.return_value.text = text


def api_error(code: int, status: str) -> errors.APIError:
    error_class = errors.ClientError if code < 500 else errors.ServerError
    return error_class(
        code, {"error": {"code": code, "message": "Upstream error", "status": status}}
    )


# ==========================================
# /health
# ==========================================
def test_health(http):
    response = http.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


# ==========================================
# Successful requests
# ==========================================
def test_summarize_returns_summary(http, mock_client):
    gemini_returns(mock_client, SUMMARY_JSON)

    response = http.post("/summarize", json={"text": VALID_TEXT})

    assert response.status_code == 200
    data = response.json()
    assert data["title"] == "Release Planning"
    assert data["key_points"] == ["The team agreed on the release scope."]
    assert data["sentiment"] == "positive"


def test_meeting_notes_returns_nested_structure(http, mock_client):
    gemini_returns(mock_client, MEETING_NOTES_JSON)

    response = http.post("/meeting-notes", json={"text": VALID_TEXT})

    assert response.status_code == 200
    data = response.json()
    assert data["risks"][0]["department"] == "Engineering"
    assert data["action_items"][0] == {"task": "Finalize wireframes.", "owner": "Sarah"}


@pytest.mark.parametrize(
    ("path", "schema", "reply"),
    [
        ("/summarize", Summary, SUMMARY_JSON),
        ("/meeting-notes", MeetingNotes, MEETING_NOTES_JSON),
    ],
    ids=["summarize", "meeting-notes"],
)
def test_each_endpoint_requests_its_own_schema(http, mock_client, path, schema, reply):
    """Each endpoint asks Gemini for its own schema, using the stripped text."""
    gemini_returns(mock_client, reply)

    http.post(path, json={"text": f"   {VALID_TEXT}   "})

    kwargs = mock_client.aio.models.generate_content.call_args.kwargs
    assert kwargs["contents"] == VALID_TEXT
    assert kwargs["config"].response_json_schema == schema.model_json_schema()


# ==========================================
# Request validation (422, before Gemini is called)
# ==========================================
@pytest.mark.parametrize(
    "body",
    [
        {},
        {"text": "too short"},
        {"text": "             "},
        {"text": "x" * 10_001},
        {"text": 12345},
    ],
    ids=["missing-text", "too-short", "only-whitespace", "too-long", "not-a-string"],
)
@pytest.mark.parametrize("path", ["/summarize", "/meeting-notes"])
def test_invalid_body_returns_422_without_calling_gemini(http, mock_client, path, body):
    response = http.post(path, json=body)

    assert response.status_code == 422
    mock_client.aio.models.generate_content.assert_not_called()


# ==========================================
# Error handlers
# ==========================================
def test_empty_prompt_returns_400(http, mock_client, monkeypatch):
    """TextRequest stops empty text first, so the guard is triggered directly."""

    async def raise_empty_prompt(*args, **kwargs):
        raise EmptyPromptError("Prompt is empty.")

    monkeypatch.setattr(api, "get_structured_async", raise_empty_prompt)

    response = http.post("/summarize", json={"text": VALID_TEXT})

    assert response.status_code == 400
    assert response.json() == {"detail": "Prompt is empty."}


@pytest.mark.parametrize("empty", ["", None], ids=["empty-string", "none"])
def test_empty_gemini_response_returns_502(http, mock_client, empty):
    gemini_returns(mock_client, empty)

    response = http.post("/summarize", json={"text": VALID_TEXT})

    assert response.status_code == 502


@pytest.mark.parametrize(
    "reply",
    [
        json.dumps({"title": "Missing the other fields"}),
        json.dumps({"title": "t", "key_points": [], "sentiment": "happy"}),
        "not json at all",
    ],
    ids=["missing-fields", "invalid-sentiment", "not-json"],
)
def test_schema_mismatch_returns_502(http, mock_client, reply):
    """Gemini's output didn't match the schema: an upstream problem, not the caller's."""
    gemini_returns(mock_client, reply)

    response = http.post("/summarize", json={"text": VALID_TEXT})

    assert response.status_code == 502
    assert "expected format" in response.json()["detail"]


def test_rate_limit_returns_429(http, mock_client):
    mock_client.aio.models.generate_content.side_effect = api_error(
        429, "RESOURCE_EXHAUSTED"
    )

    response = http.post("/summarize", json={"text": VALID_TEXT})

    assert response.status_code == 429


@pytest.mark.parametrize(
    ("code", "status"),
    [(500, "INTERNAL"), (503, "UNAVAILABLE"), (403, "PERMISSION_DENIED")],
    ids=["500", "503", "403"],
)
def test_other_api_errors_return_502(http, mock_client, code, status):
    mock_client.aio.models.generate_content.side_effect = api_error(code, status)

    response = http.post("/summarize", json={"text": VALID_TEXT})

    assert response.status_code == 502


def test_upstream_errors_do_not_leak_details(http, mock_client):
    """Gemini's raw error message is not passed back to the caller."""
    mock_client.aio.models.generate_content.side_effect = api_error(500, "INTERNAL")

    response = http.post("/summarize", json={"text": VALID_TEXT})

    assert "Upstream error" not in response.text
