import argparse
import sys

from dotenv import load_dotenv
from google import genai
from google.genai import errors

from ai_playground.clean_text import clean_text
from ai_playground.llm import ask_gemini, ask_gemini_stream, create_client


def execute_gemini(client: genai.Client, cleaned_text: str, stream: bool) -> None:
    print("\nSending request to Gemini...", file=sys.stderr)

    if stream:
        print("\nGemini Response:\n", file=sys.stderr)

        for chunk in ask_gemini_stream(client, cleaned_text):
            print(chunk, end="", flush=True)

        print()
    else:
        result = ask_gemini(client, cleaned_text)

        print("\nGemini Response:\n", file=sys.stderr)
        print(result)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize text using the Gemini API.")

    parser.add_argument(
        "--stream",
        action="store_true",
        help="Stream the response from Gemini instead of waiting for the full response.",
    )

    return parser.parse_args()


def main() -> None:
    load_dotenv()

    args = parse_arguments()

    print(
        "Paste or type your text. When finished, press Enter, then Ctrl+D (Mac/Linux) or Ctrl+Z then Enter (Windows):",
        file=sys.stderr,
    )

    try:
        raw = sys.stdin.read()
    except KeyboardInterrupt:
        print("\nError: Input cancelled!", file=sys.stderr)
        sys.exit(130)

    cleaned_text = clean_text(raw)

    if not cleaned_text:
        print("\nError: No text provided!", file=sys.stderr)
        sys.exit(1)

    client = create_client()

    try:
        execute_gemini(client, cleaned_text, args.stream)
    except errors.APIError as e:
        print(f"\nError: API request failed ({e.code}): {e.message}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nError: Request cancelled!", file=sys.stderr)
        sys.exit(130)
    finally:
        client.close()


if __name__ == "__main__":
    main()
