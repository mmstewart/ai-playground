# ai-playground

[![Tests](https://github.com/mmstewart/ai-playground/actions/workflows/tests.yml/badge.svg)](https://github.com/mmstewart/ai-playground/actions/workflows/tests.yml)

A Python command-line tool for learning AI engineering, starting with the fundamentals: packaging, typing, testing, and
CI. It cleans input text and sends it to Google's Gemini API, which summarizes it in five or fewer bullet points. It
will grow into experiments with structured output, retrieval (RAG), and evals.

## Features

- **Text cleaning:** `clean_text` trims leading/trailing whitespace and collapses runs of blank lines (including lines
  that contain only spaces or tabs) into a single blank line.
- **Summarization with Gemini:** a system instruction tells the model to be concise and summarize text in five or fewer
  bullet points, using the official `google-genai` SDK.
- **Streaming or full responses:** by default the full response prints at once; with `--stream`, it appears piece by
  piece as Gemini generates it.
- **Resilient API calls:** automatic retries with exponential backoff for transient errors (408, 429, 5xx) and a
  2-minute timeout per request.
- **Clear error handling:** API errors, empty or blocked responses, and Ctrl+C produce a readable message and a
  meaningful exit code instead of a traceback.
- **Clean output streams:** only Gemini's response goes to stdout; prompts, status messages, and errors go to stderr, so
  output can be redirected to a file or piped into other tools.
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

Run the tool, paste or type your text, press `Enter`, then `Ctrl+D` (Linux/macOS) or `Ctrl+Z` then `Enter` (Windows) to
send it:

```bash
uv run ai-playground            # wait for the full response
uv run ai-playground --stream   # stream the response as it's generated
uv run ai-playground --help     # show available options
```

Because only the response goes to stdout, you can save it straight to a file while still seeing the prompts on screen:

```bash
uv run ai-playground > summary.txt
```

Press `Ctrl+C` at any point to cancel.

### Exit codes

| Code  | Meaning                                                |
| ----- | ------------------------------------------------------ |
| `0`   | Success                                                |
| `1`   | No text provided, API error, or empty/blocked response |
| `130` | Cancelled with `Ctrl+C`                                |

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
│   ├── __init__.py               # CLI entry point: arguments, input, output
│   ├── clean_text.py             # text cleaning
│   └── llm.py                    # Gemini client setup, ask_gemini, ask_gemini_stream
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
- [x] System instruction and streaming responses
- [ ] Multi-turn chat
- [ ] Structured JSON output validated with Pydantic
- [ ] FastAPI endpoint for text extraction
- [ ] Retrieval-augmented generation (RAG) with pgvector
- [ ] Evals for answer quality
