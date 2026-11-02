# ai-playground

[![Tests](https://github.com/mmstewart/ai-playground/actions/workflows/tests.yml/badge.svg)](https://github.com/mmstewart/ai-playground/actions/workflows/tests.yml)

A Python project for learning AI engineering, starting with the fundamentals: packaging, typing, testing, and CI. It
currently cleans input text and sends it to Google's Gemini API, and will grow into experiments with structured output,
retrieval (RAG), and evals.

## Features

- **Text cleaning:** `clean_text` trims leading/trailing whitespace and collapses runs of blank lines (including lines
  that contain only spaces or tabs) into a single blank line.
- **Gemini integration:** `ask_gemini` sends a prompt to Gemini using the official `google-genai` SDK and returns the
  response text.
- **Resilient API calls:** automatic retries with exponential backoff for transient errors (408, 429, 5xx) and a
  2-minute timeout per request.
- **Clear error handling:** API errors and empty or blocked responses produce a readable message and a nonzero exit code
  instead of a traceback.
- **Testable design:** the Gemini client is created once and passed in (dependency injection), so tests use a mock and
  never call the real API.

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

Run the command-line tool, paste or type some text, then press `Ctrl+D` on a new line (Linux/macOS) or `Ctrl+Z` then
`Enter` (Windows):

```bash
uv run ai-playground
```

The text is cleaned, sent to Gemini, and the response is printed. Press `Ctrl+C` to cancel.

## Running tests

```bash
uv run pytest -v
```

Tests mock the Gemini client, so they run without an API key or network access. They also run automatically on every
push and pull request through GitHub Actions.

## Project structure

```
ai-playground/
├── .github/workflows/tests.yml   # CI: runs pytest on push and PRs
├── src/ai_playground/
│   ├── __init__.py               # CLI entry point (main)
│   ├── clean_text.py             # text cleaning
│   └── llm.py                    # Gemini client setup and ask_gemini
├── tests/
│   ├── test_clean_text.py
│   └── test_llm.py
├── .env                          # your API key (not committed)
├── pyproject.toml
└── uv.lock
```

## Roadmap

- [x] Project setup with uv, type hints, pytest, and CI
- [x] Call an LLM API with retries, timeouts, and error handling
- [ ] Structured JSON output validated with Pydantic
- [ ] FastAPI endpoint for text extraction
- [ ] Retrieval-augmented generation (RAG) with pgvector
- [ ] Evals for answer quality
