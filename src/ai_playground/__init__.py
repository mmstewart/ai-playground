import argparse
import sys

from dotenv import load_dotenv
from google import genai
from google.genai import errors
from pydantic_core import ValidationError

from ai_playground.clean_text import clean_text
from ai_playground.llm import (
    ask_gemini,
    ask_gemini_stream,
    create_chat_session,
    create_client,
    get_structured,
    send_chat_message,
)
from ai_playground.schemas import Summary


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
    mode = parser.add_mutually_exclusive_group()

    mode.add_argument(
        "--stream",
        action="store_true",
        help="Stream the response from Gemini instead of waiting for the full response.",
    )

    mode.add_argument(
        "--chat",
        action="store_true",
        help="Start a multi-turn chat instead of summarizing text.",
    )

    mode.add_argument(
        "--json",
        action="store_true",
        help="Return a structured summary (title, key points, sentiment) as JSON",
    )

    return parser.parse_args()


def run_chat(client: genai.Client) -> None:
    chat = create_chat_session(client)

    print("Chat started. Type 'quit' to exit.", file=sys.stderr)

    while True:
        try:
            message = input("\nYou: ").strip()
        except EOFError:
            print("\nGoodbye!", file=sys.stderr)
            return

        if not message:
            continue
        if message.lower() in {"quit", "exit"}:
            print("Goodbye!", file=sys.stderr)
            return

        print("\nGemini:", file=sys.stderr)
        try:
            for chunk in send_chat_message(chat, message):
                print(chunk, end="", flush=True)
            print()
        except errors.APIError as e:
            print(
                f"\nError: API request failed ({e.code}): {e.message}", file=sys.stderr
            )
        except ValueError as e:
            print(f"\nError: {e}", file=sys.stderr)


def read_input() -> str:
    print(
        "Paste or type your text. When finished, press Enter, then Ctrl+D (Mac/Linux) or Ctrl+Z then Enter (Windows):",
        file=sys.stderr,
    )

    cleaned_text = clean_text(sys.stdin.read())

    if not cleaned_text:
        print("\nError: No text provided!", file=sys.stderr)
        sys.exit(1)

    return cleaned_text


def run_single(client: genai.Client, stream: bool, as_json: bool) -> None:
    text = read_input()

    if as_json:
        summary = get_structured(client, text, Summary)
        print("\nGemini JSON Response:\n", file=sys.stderr)
        print(summary.model_dump_json(indent=2))
    else:
        execute_gemini(client, text, stream)


def main() -> None:
    load_dotenv()

    args = parse_arguments()

    client = create_client()

    try:
        if args.chat:
            run_chat(client)
        else:
            run_single(client, args.stream, args.json)
    except errors.APIError as e:
        print(f"\nError: API request failed ({e.code}): {e.message}", file=sys.stderr)
        sys.exit(1)
    except ValidationError as e:
        print(
            f"\nError: Gemini's response didn't match the expected format ({e.error_count()} issue(s)).",
            file=sys.stderr,
        )
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
