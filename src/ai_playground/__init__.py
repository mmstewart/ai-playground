import sys

from dotenv import load_dotenv
from google.genai import errors

from ai_playground.clean_text import clean_text
from ai_playground.llm import ask_gemini, create_client


def main() -> None:
    load_dotenv()

    print(
        "Paste or input text, then press Ctrl+D (Mac/Linux) or Ctrl+Z then Enter (Windows):"
    )

    try:
        raw = sys.stdin.read()
    except KeyboardInterrupt:
        print("\nInput cancelled. Exiting.")
        return

    cleaned_text = clean_text(raw)

    if not cleaned_text:
        print("No text provided. Exiting.")
        return

    client = create_client()

    try:
        result = ask_gemini(client, cleaned_text)
        print("\nGemini Response:\n")
        print(result)
    except errors.APIError as e:
        print(e.code)
        print(e.message)
        sys.exit(1)
    except ValueError as e:
        print(e)
        sys.exit(1)
    finally:
        client.close()


if __name__ == "__main__":
    main()
