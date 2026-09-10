"""Build or merge a shared tokenizer for sharded rule-model data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rl.rule_data import (
    LEGACY_CARD_ALIAS_TOKENS,
    RULE_DATA_VERSION,
    SPECIAL_TOKENS,
    RuleTokenizer,
    read_jsonl,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", nargs="*", type=Path)
    parser.add_argument("--merge", nargs="*", type=Path, default=[])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not args.data and not args.merge:
        raise ValueError("provide at least one data shard or tokenizer to merge")

    tokens: set[str] = set()
    required_tokens: set[str] = set()
    trusted_aliases: set[str] = set()
    for path in args.data:
        tokenizer = RuleTokenizer.build(read_jsonl(path))
        tokens.update(tokenizer.tokens)
        required_tokens.update(tokenizer.tokens)
    for path in args.merge:
        version = int(json.loads(path.read_text(encoding="utf-8")).get("version", -1))
        merged_tokens = set(RuleTokenizer.load(path).tokens)
        tokens.update(merged_tokens)
        if version == RULE_DATA_VERSION and not args.data:
            trusted_aliases.update(merged_tokens & LEGACY_CARD_ALIAS_TOKENS)
        elif not args.data:
            raise ValueError(
                "migrating a legacy tokenizer requires the v3 data shards so "
                "legitimate non-card token collisions can be retained"
            )

    tokens.difference_update(
        LEGACY_CARD_ALIAS_TOKENS - required_tokens - trusted_aliases
    )

    # Match RuleTokenizer.build/canonicalize_training_tokenizer exactly. Numeric
    # tokens belong in the sorted vocabulary; only decoder-control specials have
    # fixed leading ids.
    special_set = set(SPECIAL_TOKENS)
    tokenizer = RuleTokenizer((
        *SPECIAL_TOKENS, *sorted(tokens - special_set),
    ))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(args.out)
    print(f"[rule-tokenizer] vocab={len(tokenizer.tokens)} out={args.out.resolve()}")


if __name__ == "__main__":
    main()
