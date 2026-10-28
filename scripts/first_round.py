"""Run one DeepSeek Harness turn to confirm the Python SDK can reach dsh."""

import os
import sys
from pathlib import Path

from deepseek_harness import DeepSeekHarness

ROOT = Path(__file__).resolve().parents[1]


def load_env(path: Path) -> None:
    if not path.is_file():
        return
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def main() -> int:
    load_env(ROOT / ".env")
    if not os.environ.get("DEEPSEEK_API_KEY"):
        print("DEEPSEEK_API_KEY is missing", file=sys.stderr)
        return 1

    dsh_home = ROOT / ".dsh-home"
    dsh_home.mkdir(parents=True, exist_ok=True)

    with DeepSeekHarness(
        dsh_home=str(dsh_home),
        cwd=str(ROOT),
        provider="deepseek-official",
        model="deepseek-v4-flash",
        profile="sdk-minimal",
    ) as harness:
        result = harness.run(
            "Reply with exactly this token and nothing else: harness-ok",
            session_id="first-round",
        )

    print(result.final_response)
    print(f"finish_reason={result.finish_reason}")
    return 0 if result.finish_reason == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
