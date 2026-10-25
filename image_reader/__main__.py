"""Command-line entry point; conversation history lives only in this process."""

import argparse
import os
from pathlib import Path
import sys

from .client import Conversation, ImageReaderError, OllamaClient


def main(argv=None):
    parser = argparse.ArgumentParser(description="Ask questions about local images or extract Markdown text using Ollama.")
    parser.add_argument("--endpoint", default=os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434"))
    parser.add_argument("--model", default="llama3.2-vision")
    parser.add_argument("--timeout", type=float, default=120)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("ask", "ocr", "chat"):
        command = sub.add_parser(name)
        command.add_argument("--image", action="append", required=True, help="Local PNG/JPEG/WebP; repeat for multiple images")
        command.add_argument("--system")
        if name != "ocr":
            command.add_argument("prompt")
        if name != "chat":
            command.add_argument("--output", type=Path, help="Create a new UTF-8 answer file (existing files are preserved)")
    args = parser.parse_args(argv)
    try:
        if getattr(args, "output", None) and args.output.exists():
            raise ImageReaderError("Output file already exists; choose a new path.")
        session = Conversation(OllamaClient(args.endpoint, args.model, args.timeout), args.system)
        answer = session.ocr(args.image) if args.command == "ocr" else session.ask(args.prompt, args.image)
        if getattr(args, "output", None):
            with args.output.open("x", encoding="utf-8") as stream:
                stream.write(answer + "\n")
        print(answer)
        if args.command == "chat":
            print("\nFollow-up text (images retained in memory); /quit exits.", file=sys.stderr)
            while True:
                try:
                    prompt = input("you> ")
                except EOFError:
                    break
                if prompt.strip() == "/quit":
                    break
                if not prompt.strip():
                    continue
                print(session.ask(prompt))
        return 0
    except (ImageReaderError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
