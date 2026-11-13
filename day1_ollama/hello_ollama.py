"""
Day 1 / Task 4: Send a prompt to a local LLM served by Ollama and print the response.

Usage (from the project root, with the `local-rag-api` conda env activated):
    python day1_ollama/hello_ollama.py
    python day1_ollama/hello_ollama.py "Your own question here"
"""

import sys
import time

from ollama import ChatResponse, ResponseError, chat

# Must match the "name:tag" used with `ollama pull`.
MODEL = "llama3.2:3b"

# Used when no prompt is passed on the command line.
DEFAULT_PROMPT = "Explain what a local LLM is in 2 sentences."


def ns_to_sec(ns: int | None) -> float:
    """Convert nanoseconds (Ollama's time unit) to seconds."""
    return (ns or 0) / 1_000_000_000


def main() -> None:
    # Join all CLI arguments into one prompt; fall back to the default if none are given.
    prompt = " ".join(sys.argv[1:]) or DEFAULT_PROMPT

    # Chat-style input: a list of {role, content} messages.
    # The model is stateless, so any conversation history must be sent here every time.
    messages = [{"role": "user", "content": prompt}]

    print(f"Model : {MODEL}")
    print(f"Prompt: {prompt}\n")

    # Measure end-to-end latency from the caller's point of view.
    start = time.perf_counter()
    try:
        # Sends an HTTP request to the Ollama server at localhost:11434.
        # The model runs inside the Ollama app; this library is only a client.
        response: ChatResponse = chat(model=MODEL, messages=messages)
    except ConnectionError:
        # Raised when the Ollama app (server) is not running.
        print("Error: Could not connect to Ollama. Please start the Ollama app.")
        sys.exit(1)
    except ResponseError as e:
        # Raised by the server, e.g. when the model name is wrong or not pulled yet.
        print(f"Error from Ollama (status {e.status_code}): {e.error}")
        sys.exit(1)
    elapsed = time.perf_counter() - start

    print("Response:")
    print(response.message.content)

    # Same metrics as `ollama run --verbose`, read from the response object.
    eval_count = response.eval_count or 0
    eval_sec = ns_to_sec(response.eval_duration)
    tokens_per_sec = eval_count / eval_sec if eval_sec > 0 else 0.0

    print("\n--- Stats ---")
    print(f"load duration    : {ns_to_sec(response.load_duration):.2f} s")
    print(f"prompt tokens    : {response.prompt_eval_count}")
    print(f"generated tokens : {eval_count}")
    print(f"eval rate        : {tokens_per_sec:.2f} tokens/s")
    print(f"wall-clock time  : {elapsed:.2f} s (measured in Python)")


# Run main() only when this file is executed directly, not when it is imported.
if __name__ == "__main__":
    main()