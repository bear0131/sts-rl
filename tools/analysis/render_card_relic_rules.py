"""Export every currently supported card and combat-relic rule on one line.

The output is a plain UTF-8 text file. Every non-empty line is one canonical
rule, with tab-separated metadata followed by the rule sentence::

    CARD<TAB>card id<TAB>base|upgraded<TAB>rule sentence
    RELIC<TAB>relic id<TAB>rule sentence

Card ids are labels for this human-readable catalog only. The model still
receives anonymous rule-token programs. The current rule source covers the
Ironclad/status language cards and combat relics; other character cards and
extra curse-only rules are excluded.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from env.core.card_language import LANGUAGE_CARDS
from env.core.rule_language import (
    COMBAT_RELIC_RULES,
    card_rule_sentence,
    relic_rule_sentence,
)


def iter_rule_lines():
    # LANGUAGE_CARDS is the supported Ironclad + status-card surface. Other
    # character cards and the separate extra-curse table are intentionally
    # outside this catalog.
    for card_id in sorted(LANGUAGE_CARDS):
        for upgrades, version in ((0, "base"), (1, "upgraded")):
            yield "\t".join((
                "CARD", card_id, version,
                card_rule_sentence(card_id, upgrades),
            ))
    for relic_id in sorted(COMBAT_RELIC_RULES):
        yield "\t".join(("RELIC", relic_id, relic_rule_sentence(relic_id)))


def render_rules() -> str:
    return "\n".join(iter_rule_lines()) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path,
        default=Path("artifacts/local/card-relic-rule-descriptions-v6.txt"),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    lines = list(iter_rule_lines())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {len(lines)} one-rule lines to {args.output}")


if __name__ == "__main__":
    main()
