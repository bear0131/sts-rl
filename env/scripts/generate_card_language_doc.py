"""Regenerate the bilingual Ironclad and Status card-language reference."""

from __future__ import annotations

import argparse
from pathlib import Path

from env.core.card_language import render_markdown


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("doc/ironclad_card_language.md"),
    )
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_markdown() + "\n", encoding="utf-8")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
