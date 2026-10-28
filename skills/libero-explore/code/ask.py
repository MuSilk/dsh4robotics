#!/usr/bin/env python3
"""Ask imajev one question about one or two images and print the whole body.

    python3 code/ask.py obj06.png obj05.png

Use this before wiring imajev into a controller: it proves the socket, the image
path rule, and the response field names without spending a robot step.  The first
call of a session includes the model load (~2-3 min), later ones are seconds.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vlm  # noqa: E402

QUESTION = {"which": {
    "type": "choice",
    "instructions": "Which of these crops shows a pack of butter - a small box whose "
                    "label says BUTTER and is mostly warm yellow/orange?",
    "criteria": {"first": "the FIRST picture is butter",
                 "second": "the SECOND picture is butter",
                 "neither": "neither picture shows butter"}}}


def main():
    images = sys.argv[1:]
    print("available:", vlm.available(), flush=True)
    try:
        body = vlm.decide(QUESTION, images=images)
    except Exception as exc:
        print("decide failed:", type(exc).__name__, exc, flush=True)
        return 1
    print(json.dumps(body, indent=2)[:4000], flush=True)
    print("parsed:", vlm.answer(body, "which"), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
