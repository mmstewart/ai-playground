import io
import json
import sys
from unittest.mock import MagicMock

import pytest
from google.genai import errors

import ai_playground
from ai_playground import main


# ==========================================
# Helpers and fixtures
# ==========================================
def make_chunk(text: str | None) -> MagicMock:
    """Build a fake stream chunk with the given text."""
    chunk = MagicMock()
    chunk.text = text
    return chunk


@pytest.fixture
def mock_client(monkeypatch) -> MagicMock:
    """Replace create_client in main's module with one returning a fake client.

    Patched on ai_playground (where main looks it up), not ai_playground.llm.
    """
    client = MagicMock()
    monkeypatch.setattr(ai_playground, "create_client", lambda: client)
    return client


@pytest.fixture(autouse=True)
def no_dotenv(monkeypatch):
    """Don't read the real .env during tests."""
    monkeypatch.setattr(ai_playground, "load_dotenv", lambda: None)


def run_main(monkeypatch, stdin_text: str = "", args: list[str] | None = None) -> None:
    """Run main() with fake command-line arguments and fake stdin."""
    monkeypatch.setattr(sys, "argv", ["ai-playground", *(args or [])])
    monkeypatch.setattr(sys, "stdin", io.StringIO(stdin_text))
    main()


# ==========================================
# Successful runs
# ==========================================
def test_full_response_goes_to_stdout(monkeypatch, capsys, mock_client):
    """Without --stream, the full response prints to stdout; labels go to stderr."""
    mock_client.models.generate_content.return_value.text = "- A short summary"

    run_main(monkeypatch, "Some text to summarize\n")

    captured = capsys.readouterr()
    assert captured.out == "- A short summary\n"
    assert "Gemini Response" in captured.err
    assert "Paste or type your text" in captured.err


def test_stream_prints_chunks_to_stdout(monkeypatch, capsys, mock_client):
    """With --stream, chunks print in order, followed by one newline."""
    mock_client.models.generate_content_stream.return_value = [
        make_chunk("Hello"),
        make_chunk(None),
        make_chunk(" world"),
    ]

    run_main(monkeypatch, "Some text\n", ["--stream"])

    captured = capsys.readouterr()
    assert captured.out == "Hello world\n"
    mock_client.models.generate_content.assert_not_called()


def test_input_is_cleaned_before_sending(monkeypatch, mock_client):
    """The text sent to Gemini has been run through clean_text."""
    mock_client.models.generate_content.return_value.text = "ok"

    run_main(monkeypatch, "   first\n\n\n\nsecond   \n")

    sent = mock_client.models.generate_content.call_args.kwargs["contents"]
    assert sent == "first\n\nsecond"


def test_client_closed_after_success(monkeypatch, mock_client):
    mock_client.models.generate_content.return_value.text = "ok"

    run_main(monkeypatch, "Some text\n")

    mock_client.close.assert_called_once()


# ==========================================
# Early exits (before any API call)
# ==========================================
@pytest.mark.parametrize("stdin_text", ["", "   \n\n\t\n"])
def test_empty_input_exits_1_without_calling_api(
    monkeypatch, capsys, mock_client, stdin_text
):
    with pytest.raises(SystemExit) as exc_info:
        run_main(monkeypatch, stdin_text)

    assert exc_info.value.code == 1
    captured = capsys.readouterr()
    assert "No text provided" in captured.err
    assert captured.out == ""
    mock_client.models.generate_content.assert_not_called()


def test_ctrl_c_during_input_exits_130(monkeypatch, capsys, mock_client):
    """Ctrl+C while typing exits with 130 and no traceback."""
    fake_stdin = MagicMock()
    fake_stdin.read.side_effect = KeyboardInterrupt
    monkeypatch.setattr(sys, "argv", ["ai-playground"])
    monkeypatch.setattr(sys, "stdin", fake_stdin)

    with pytest.raises(SystemExit) as exc_info:
        main()

    assert exc_info.value.code == 130
    assert "cancelled" in capsys.readouterr().err.lower()
    mock_client.models.generate_content.assert_not_called()


def test_help_exits_before_reading_input(monkeypatch, capsys):
    """--help prints usage and exits 0 without waiting for stdin."""
    fake_stdin = MagicMock()
    monkeypatch.setattr(sys, "argv", ["ai-playground", "--help"])
    monkeypatch.setattr(sys, "stdin", fake_stdin)

    with pytest.raises(SystemExit) as exc_info:
        main()

    assert exc_info.value.code == 0
    assert "--stream" in capsys.readouterr().out
    fake_stdin.read.assert_not_called()


def test_unknown_flag_exits_2(monkeypatch):
    """argparse rejects unknown flags with exit code 2."""
    monkeypatch.setattr(sys, "argv", ["ai-playground", "--strem"])

    with pytest.raises(SystemExit) as exc_info:
        main()

    assert exc_info.value.code == 2


# ==========================================
# Errors during the request
# ==========================================
@pytest.mark.parametrize("args", [[], ["--stream"]], ids=["full", "stream"])
def test_api_error_exits_1_and_closes_client(monkeypatch, capsys, mock_client, args):
    error = errors.ServerError(
        503,
        {"error": {"code": 503, "message": "High demand", "status": "UNAVAILABLE"}},
    )
    mock_client.models.generate_content.side_effect = error
    mock_client.models.generate_content_stream.side_effect = error

    with pytest.raises(SystemExit) as exc_info:
        run_main(monkeypatch, "Some text\n", args)

    assert exc_info.value.code == 1
    captured = capsys.readouterr()
    assert "503" in captured.err
    assert captured.out == ""
    mock_client.close.assert_called_once()


def test_empty_response_exits_1(monkeypatch, capsys, mock_client):
    mock_client.models.generate_content.return_value.text = None

    with pytest.raises(SystemExit) as exc_info:
        run_main(monkeypatch, "Some text\n")

    assert exc_info.value.code == 1
    assert "empty response" in capsys.readouterr().err
    mock_client.close.assert_called_once()


def test_ctrl_c_during_request_exits_130_and_closes_client(
    monkeypatch, capsys, mock_client
):
    mock_client.models.generate_content.side_effect = KeyboardInterrupt

    with pytest.raises(SystemExit) as exc_info:
        run_main(monkeypatch, "Some text\n")

    assert exc_info.value.code == 130
    assert "cancelled" in capsys.readouterr().err.lower()
    mock_client.close.assert_called_once()


# ==========================================
# Chat mode (--chat)
# ==========================================
FakeInput = str | BaseException | type[BaseException]


def fake_inputs(monkeypatch, items: list[FakeInput]) -> None:
    """Replace input() with one that returns items in order.

    An exception class or instance in the list is raised instead of returned,
    which simulates Ctrl+D (EOFError) or Ctrl+C (KeyboardInterrupt).
    """
    remaining = iter(items)

    def fake_input(prompt: str = "") -> str:
        item = next(remaining)
        if isinstance(item, str):
            return item
        raise item

    monkeypatch.setattr("builtins.input", fake_input)


def run_chat_main(monkeypatch, inputs: list[FakeInput]) -> MagicMock:
    """Run main() in chat mode. Returns the fake stdin so tests can check it wasn't read."""
    fake_inputs(monkeypatch, inputs)
    fake_stdin = MagicMock()
    monkeypatch.setattr(sys, "argv", ["ai-playground", "--chat"])
    monkeypatch.setattr(sys, "stdin", fake_stdin)
    main()
    return fake_stdin


@pytest.fixture
def mock_chat(mock_client) -> MagicMock:
    """The fake chat session that create_chat_session will return."""
    return mock_client.chats.create.return_value


def test_chat_reuses_one_session_for_all_messages(
    monkeypatch, capsys, mock_client, mock_chat
):
    """History works only if the chat is created once and reused for every message."""
    mock_chat.send_message_stream.side_effect = [
        [make_chunk("Rayleigh scattering.")],
        [make_chunk("Air bounces blue light around.")],
    ]

    run_chat_main(
        monkeypatch,
        ["why is the sky blue?", "explain that to a 5 year old", "quit"],
    )

    mock_client.chats.create.assert_called_once()
    sent = [call.args[0] for call in mock_chat.send_message_stream.call_args_list]
    assert sent == ["why is the sky blue?", "explain that to a 5 year old"]

    out = capsys.readouterr().out
    assert "Rayleigh scattering." in out
    assert "Air bounces blue light around." in out


@pytest.mark.parametrize("command", ["quit", "exit", "QUIT", "  Exit  "])
def test_chat_quit_commands_end_cleanly(
    monkeypatch, capsys, mock_client, mock_chat, command
):
    """quit/exit end the chat without an error exit, case-insensitively."""
    run_chat_main(monkeypatch, [command])

    mock_chat.send_message_stream.assert_not_called()
    assert "goodbye" in capsys.readouterr().err.lower()
    mock_client.close.assert_called_once()


def test_chat_skips_empty_lines(monkeypatch, mock_chat):
    mock_chat.send_message_stream.return_value = [make_chunk("Hi!")]

    run_chat_main(monkeypatch, ["", "   ", "hello", "quit"])

    mock_chat.send_message_stream.assert_called_once_with("hello")


def test_chat_ctrl_d_says_goodbye(monkeypatch, capsys, mock_client):
    """Ctrl+D (EOFError) ends the chat cleanly."""
    run_chat_main(monkeypatch, [EOFError])

    assert "goodbye" in capsys.readouterr().err.lower()
    mock_client.close.assert_called_once()


def test_chat_ctrl_c_exits_130_and_closes_client(monkeypatch, capsys, mock_client):
    with pytest.raises(SystemExit) as exc_info:
        run_chat_main(monkeypatch, [KeyboardInterrupt])

    assert exc_info.value.code == 130
    assert "cancelled" in capsys.readouterr().err.lower()
    mock_client.close.assert_called_once()


def test_chat_continues_after_api_error(monkeypatch, capsys, mock_chat):
    """One failed message is reported, and the next message still works."""
    error = errors.ServerError(
        503,
        {"error": {"code": 503, "message": "High demand", "status": "UNAVAILABLE"}},
    )
    mock_chat.send_message_stream.side_effect = [
        error,
        [make_chunk("Second answer")],
    ]

    run_chat_main(monkeypatch, ["first", "second", "quit"])

    assert mock_chat.send_message_stream.call_count == 2
    captured = capsys.readouterr()
    assert "503" in captured.err
    assert "Second answer" in captured.out


def test_chat_continues_after_empty_response(monkeypatch, capsys, mock_chat):
    mock_chat.send_message_stream.side_effect = [
        [make_chunk(None)],
        [make_chunk("Second answer")],
    ]

    run_chat_main(monkeypatch, ["first", "second", "quit"])

    captured = capsys.readouterr()
    assert "empty response" in captured.err
    assert "Second answer" in captured.out


def test_chat_mode_does_not_read_stdin(monkeypatch, capsys, mock_client):
    """Chat mode uses input(), not the paste-and-Ctrl+D prompt."""
    fake_stdin = run_chat_main(monkeypatch, ["quit"])

    fake_stdin.read.assert_not_called()
    assert "Paste or type your text" not in capsys.readouterr().err


# ==========================================
# Structured output (--json)
# ==========================================
def test_json_prints_valid_summary_to_stdout(monkeypatch, capsys, mock_client):
    """--json prints only valid JSON to stdout, so it can be saved or piped."""
    mock_client.models.generate_content.return_value.text = json.dumps(
        {
            "title": "Why the Sky Is Blue",
            "key_points": ["Blue light scatters more."],
            "sentiment": "neutral",
        }
    )

    run_main(monkeypatch, "Why is the sky blue?\n", ["--json"])

    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert data["title"] == "Why the Sky Is Blue"
    assert data["key_points"] == ["Blue light scatters more."]
    assert data["sentiment"] == "neutral"
    mock_client.models.generate_content_stream.assert_not_called()
    mock_client.close.assert_called_once()


def test_json_validation_error_exits_1(monkeypatch, capsys, mock_client):
    """A response that doesn't match the schema gives a short error, not a traceback."""
    mock_client.models.generate_content.return_value.text = json.dumps(
        {"title": "Missing fields"}
    )

    with pytest.raises(SystemExit) as exc_info:
        run_main(monkeypatch, "Some text\n", ["--json"])

    assert exc_info.value.code == 1
    captured = capsys.readouterr()
    assert "expected format" in captured.err
    assert captured.out == ""
    mock_client.close.assert_called_once()


@pytest.mark.parametrize(
    "args",
    [
        ["--json", "--stream"],
        ["--json", "--chat"],
        ["--chat", "--stream"],
        ["--json", "--stream", "--chat"],
    ],
    ids=["json-stream", "json-chat", "chat-stream", "all-three"],
)
def test_conflicting_flags_exit_2(monkeypatch, capsys, args):
    """--json, --stream, and --chat can't be combined."""
    fake_stdin = MagicMock()
    monkeypatch.setattr(sys, "argv", ["ai-playground", *args])
    monkeypatch.setattr(sys, "stdin", fake_stdin)

    with pytest.raises(SystemExit) as exc_info:
        main()

    assert exc_info.value.code == 2
    assert "not allowed with" in capsys.readouterr().err
    fake_stdin.read.assert_not_called()
