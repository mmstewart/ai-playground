import argparse
import asyncio
import sys
import time

from google import genai

from ai_playground.clean_text import clean_text
from ai_playground.llm import (
    ask_gemini,
    ask_gemini_async,
    create_client,
)

TEXTS = [
    "The quick brown fox jumps over the lazy dog.",
    "In a distant future, humanity has colonized the stars, but a mysterious alien race threatens to destroy everything.",
    "Once upon a time, in a land far away, there lived a brave knight who embarked on a quest to save the kingdom from an evil sorcerer.",
    "In the heart of the bustling city, a young artist struggles to find inspiration amidst the chaos and noise, until one day, a chance encounter changes everything.",
    "The sun sets over the horizon, casting a warm golden glow across the landscape, as the gentle breeze carries the scent of blooming flowers and the promise of a new day.",
    (
        "The city council met on Tuesday to discuss the proposed light rail extension. "
        "Supporters argued it would cut commute times by up to 25 minutes for residents in the northern suburbs "
        "and reduce traffic on the main highway. Opponents raised concerns about the $1.2 billion cost, "
        "two years of construction disruption on Main Street, and the loss of 300 parking spaces. "
        "Several small business owners said they feared losing customers during construction. "
        "The council agreed to hold two more public hearings and commission an independent cost review "
        "before a final vote in March."
    ),
    (
        "Our team migrated the billing service from a monolith to three separate services over six weeks. "
        "Response times improved by 40 percent, but we hit problems with duplicate invoices when two services "
        "processed the same event. We fixed it by adding idempotency keys to every payment request. "
        "Deployment also became more complex, so we added health checks and a staged rollout. "
        "Next quarter, we plan to move the reporting module, add better monitoring, and write a runbook "
        "for on-call engineers covering the most common failure scenarios."
    ),
    (
        "Photosynthesis is the process plants use to turn light into chemical energy. "
        "It happens mainly in the chloroplasts, which contain chlorophyll, the pigment that absorbs red and blue light "
        "and reflects green. In the light-dependent reactions, water molecules are split, releasing oxygen as a byproduct. "
        "The energy captured is stored in ATP and NADPH. In the Calvin cycle, the plant uses that energy to convert "
        "carbon dioxide from the air into glucose, which it uses for growth or stores as starch for later."
    ),
    (
        "I bought these wireless headphones three months ago. The sound quality is excellent, especially the bass, "
        "and noise cancellation works well on flights. Battery life is about 28 hours, close to the advertised 30. "
        "However, the ear cushions started cracking after two months, the touch controls often register accidental taps, "
        "and the companion app crashes when updating firmware. Customer support replaced the cushions for free, "
        "which was appreciated, but at this price I expected better build quality. I'd rate them three out of five."
    ),
    (
        "The library will launch a free coding program for teenagers this summer. Sessions run twice a week for eight weeks "
        "and cover web basics, Python, and building a small project from start to finish. Laptops will be provided, "
        "so no equipment is needed. Volunteer mentors from local tech companies will lead each group of six students. "
        "Registration opens May 1 and is limited to 48 spots, with priority for students who don't have a computer at home. "
        "The program is funded by a community grant, and organizers hope to expand it to other branches next year."
    ),
]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark Gemini API requests.")

    parser.add_argument(
        "--mode",
        choices=["sequential", "concurrent"],
        required=True,
        help="Run the requests one at a time (sequential) or all at once (concurrent).",
    )

    return parser.parse_args()


def run_sequential(client: genai.Client, texts: list[str]) -> None:
    time_start = time.perf_counter()

    for text in texts:
        cleaned_text = clean_text(text)
        ask_gemini(client, cleaned_text)

    time_end = time.perf_counter()

    print(f"\nSequential - {len(texts)} requests:", file=sys.stderr)
    print(f"Response time: {time_end - time_start:.2f} seconds", file=sys.stderr)


async def run_concurrent(client: genai.Client, texts: list[str]) -> None:
    tasks = []

    for text in texts:
        cleaned_text = clean_text(text)
        tasks.append(ask_gemini_async(client, cleaned_text))

    time_start = time.perf_counter()

    results = await asyncio.gather(*tasks, return_exceptions=True)

    time_end = time.perf_counter()

    failures = [result for result in results if isinstance(result, BaseException)]

    print(f"\nConcurrent - {len(texts)} requests:", file=sys.stderr)
    print(f"Response time: {time_end - time_start:.2f} seconds", file=sys.stderr)

    if failures:
        print(f"Failed: {len(failures)} of {len(texts)}", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)


def main() -> None:
    args = parse_arguments()

    client = create_client()

    try:
        print(
            f"Running {len(TEXTS)} requests ({args.mode})...",
            file=sys.stderr,
        )

        if args.mode == "sequential":
            print("\nRunning sequential requests...", file=sys.stderr)
            run_sequential(client, TEXTS)
        elif args.mode == "concurrent":
            print("\nRunning concurrent requests...", file=sys.stderr)
            asyncio.run(run_concurrent(client, TEXTS))
    finally:
        client.close()


if __name__ == "__main__":
    main()
