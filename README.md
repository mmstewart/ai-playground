# ai-playground

[![Tests](https://github.com/mmstewart/ai-playground/actions/workflows/tests.yml/badge.svg)](https://github.com/mmstewart/ai-playground/actions/workflows/tests.yml)

A Python project for learning AI engineering, starting with the fundamentals: packaging, typing, testing, and CI. It
will grow into experiments with LLM APIs, retrieval (RAG), and evals.

## Features

- `clean_text`: normalizes messy input text by trimming leading/trailing whitespace and collapsing runs of blank lines
  (including lines that contain only spaces or tabs) into a single blank line.

```python
from ai_playground import clean_text

clean_text("  hello\n\n\n\nworld  ")
# 'hello\n\nworld'
```

## Requirements

- [uv](https://docs.astral.sh/uv/) (installs the correct Python version automatically)

## Setup

```bash
git clone https://github.com/mmstewart/ai-playground.git
cd ai-playground
uv sync
```

## Usage

Run the command-line tool, paste or type some text, then press `Ctrl+D` on a new line (Linux/macOS) or `Ctrl+Z` then
`Enter` (Windows):

```bash
uv run ai-playground
```

## Running tests

```bash
uv run pytest -v
```

Tests also run automatically on every push and pull request through GitHub Actions.

## Project structure

```
ai-playground/
├── .github/workflows/tests.yml   # CI: runs pytest on push and PRs
├── src/ai_playground/
│   └── __init__.py               # clean_text and the CLI entry point
├── tests/
│   └── test_clean_text.py
├── pyproject.toml
└── uv.lock
```

## Roadmap

- [x] Project setup with uv, type hints, pytest, and CI (typed besides readme)
- [ ] Call an LLM API with structured output
- [ ] FastAPI endpoint for text extraction
- [ ] Retrieval-augmented generation (RAG) with pgvector
- [ ] Evals for answer quality
