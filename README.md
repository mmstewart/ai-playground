# ai-playground

[![Tests](https://github.com/mmstewart/ai-playground/actions/workflows/tests.yml/badge.svg)](https://github.com/mmstewart/ai-playground/actions/workflows/tests.yml)
[![Lint](https://github.com/mmstewart/ai-playground/actions/workflows/lint.yml/badge.svg)](https://github.com/mmstewart/ai-playground/actions/workflows/lint.yml)
[![Type Check](https://github.com/mmstewart/ai-playground/actions/workflows/typecheck.yml/badge.svg)](https://github.com/mmstewart/ai-playground/actions/workflows/typecheck.yml)

A Python command-line tool for learning AI engineering, starting with the fundamentals: packaging, typing, testing, and
CI. It works with Google's Gemini API in three modes: summarizing pasted text in five or fewer bullet points, returning
a structured summary as validated JSON, or holding a multi-turn chat that remembers earlier messages. It will grow into
experiments with async requests, retrieval (RAG), and evals.

## Features

- **Text cleaning:** `clean_text` trims leading/trailing whitespace and collapses runs of blank lines (including lines
  that contain only spaces or tabs) into a single blank line.
- **Summarization:** a system instruction tells Gemini to summarize text in five or fewer bullet points, using the
  official `google-genai` SDK.
- **Structured output:** `--json` asks Gemini for a summary matching a Pydantic schema (title, up to five key points,
  and a sentiment of positive, neutral, or negative), validates the response, and prints it as JSON. A generic
  `get_structured` function works with any Pydantic model.
- **Multi-turn chat:** `--chat` starts a conversation with its own system instruction. One chat session is reused for
  every message, so Gemini keeps the conversation history.
- **Streaming or full responses:** summaries print all at once by default, or piece by piece with `--stream`. Chat
  replies always stream.
- **Resilient API calls:** automatic retries with exponential backoff for transient errors (408, 429, 5xx) and a
  2-minute timeout per request.
- **Clear error handling:** API errors, empty or blocked responses, and Ctrl+C produce a readable message and a
  meaningful exit code instead of a traceback. In chat mode, a failed message is reported and the conversation
  continues.
- **Clean output streams:** only Gemini's responses go to stdout; prompts, status messages, and errors go to stderr, so
  output can be redirected to a file or piped into other tools.
- **Testable design:** the Gemini client is created once and passed in (dependency injection), so tests use mocks and
  never call the real API.
- **Code quality in CI:** pytest, Ruff (lint and formatting), and Pyright (type checking) run on every push.

```python
from ai_playground.clean_text import clean_text

clean_text("  hello\n\n   \n\nworld  ")
# 'hello\n\nworld'
```

## Requirements

- [uv](https://docs.astral.sh/uv/) (installs the correct Python version automatically)
- A Gemini API key from [Google AI Studio](https://aistudio.google.com/) (the free tier works)

## Setup

```bash
git clone https://github.com/mmstewart/ai-playground.git
cd ai-playground
uv sync
```

Create a `.env` file in the project root with your API key:

```
GEMINI_API_KEY=your-key-here
```

`.env` is listed in `.gitignore`, so the key is never committed.

## Usage

### Summarize text

Run the tool, paste or type your text, press `Enter`, then `Ctrl+D` (Linux/macOS) or `Ctrl+Z` then `Enter` (Windows) to
send it:

```bash
uv run ai-playground            # wait for the full summary
uv run ai-playground --stream   # stream the summary as it's generated
uv run ai-playground --help     # show available options
```

Because only the response goes to stdout, you can save it straight to a file while still seeing the prompts on screen:

```bash
uv run ai-playground > summary.txt
```

### Structured JSON summary

```bash
uv run ai-playground --json
uv run ai-playground --json > summary.json   # save valid JSON only
```

The response is validated before it's printed, so stdout always contains JSON in this shape:

```json
{
	"title": "Why the Sky Is Blue",
	"key_points": [
		"Sunlight contains all colors of visible light.",
		"Shorter blue wavelengths scatter more in the atmosphere."
	],
	"sentiment": "neutral"
}
```

If Gemini's response doesn't match the schema, the tool prints a short error and exits with code 1.

### Chat

```bash
uv run ai-playground --chat
```

Type a message and press `Enter` to send it. Follow-up questions build on earlier answers. Type `quit` or `exit`, or
press `Ctrl+D`, to end the chat.

`--stream`, `--json`, and `--chat` are separate modes and can't be combined. Press `Ctrl+C` at any point to cancel.

### Exit codes

| Code  | Meaning                                                                                               |
| ----- | ----------------------------------------------------------------------------------------------------- |
| `0`   | Success, or chat ended normally                                                                       |
| `1`   | No text provided, API error, empty/blocked response, or a response that doesn't match the JSON schema |
| `2`   | Invalid or conflicting command-line options                                                           |
| `130` | Cancelled with `Ctrl+C`                                                                               |

## Development

```bash
uv run pytest -v          # tests
uv run ruff check         # lint
uv run ruff format        # format code
uv run pyright            # type check
```

Tests mock the Gemini client and chat session, so they run without an API key or network access. All checks run
automatically on every push and pull request through GitHub Actions.

## Project structure

```
ai-playground/
├── .github/workflows/
│   ├── tests.yml                 # pytest
│   ├── lint.yml                  # Ruff lint and format check
│   └── typecheck.yml             # Pyright
├── src/ai_playground/
│   ├── __init__.py               # CLI: arguments, input, summary, JSON, and chat modes
│   ├── clean_text.py             # text cleaning
│   ├── llm.py                    # Gemini client, instructions, summary, structured, and chat calls
│   └── schemas.py                # Pydantic models for structured output
├── tests/
│   ├── test_clean_text.py
│   ├── test_llm.py
│   └── test_main.py
├── .env
├── pyproject.toml
└── uv.lock
```

## Roadmap

- [x] Project setup with uv, type hints, pytest, and CI
- [x] Call an LLM API with retries, timeouts, and error handling
- [x] System instructions and streaming responses
- [x] Multi-turn chat
- [x] Linting, formatting, and type checking in CI
- [x] Structured JSON output validated with Pydantic
- [ ] Async requests
- [ ] FastAPI endpoint for text extraction
- [ ] Retrieval-augmented generation (RAG) with pgvector
- [ ] Evals for answer quality
