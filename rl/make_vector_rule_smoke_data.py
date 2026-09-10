"""Create a tiny schema-v5 vector-rule dataset for pipeline smoke tests."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import sys

from rl.rule_vector_model import CARD_RULE_VOCAB, StateVectorConfig
from rl.vector_rule_data import VECTOR_DATA_SCHEMA_VERSION


def build_record(index: int, config: StateVectorConfig, rng: random.Random) -> dict:
    player = [0.0] * config.player_dim
    player[config.player_scalar_names.index("energy")] = 0.3
    enemies = [[0.0] * config.enemy_dim for _ in range(config.max_enemies)]
    enemies[0][config.enemy_scalar_names.index("present")] = 1.0
    enemies[0][config.enemy_scalar_names.index("hp")] = 0.20 + 0.01 * (index % 6)
    piles = [0.02 + 0.01 * (index % 3), 0.01 + 0.01 * ((index // 3) % 3), 0.0]
    delta = [0.0] * config.world_dim
    delta[config.player_scalar_names.index("energy")] = -0.1
    discard_offset = config.player_dim + config.max_enemies * config.enemy_dim
    delta[discard_offset + config.pile_names.index("discard_count")] = 0.01
    offset_seed = rng.randrange(max(1, len(CARD_RULE_VOCAB) - 1))
    action_program = [1 + (offset_seed + index + offset) % max(1, len(CARD_RULE_VOCAB) - 1) for offset in range(6)]
    return {
        "schema_version": VECTOR_DATA_SCHEMA_VERSION,
        "state": {
            "player": player,
            "enemies": enemies,
            "piles": piles,
            "relic_counters": [],
        },
        "action": {"card_rule": action_program, "upgrade": index % 2},
        "hand": [
            {"card_rule": [1, 2, 3], "upgrade": 0},
            {"card_rule": [2, 3, 4], "upgrade": index % 2},
        ],
        "relics": [],
        "events": [
            {
                "kind": "EVENT_CARD_PLAY",
                "entity_ref": "ENTITY_SELF",
                "delta": delta,
            },
            {
                "kind": "EVENT_PILE_DELTA",
                "entity_ref": "ENTITY_PILES",
                "delta": [
                    value if position == discard_offset + config.pile_names.index("discard_count") else 0.0
                    for position, value in enumerate(delta)
                ],
            },
            {
                "kind": "EVENT_ACTION_COMPLETE",
                "entity_ref": "ENTITY_SELF",
            },
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("rl/data/vector-rule-smoke-v6.jsonl"))
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--stdout", action="store_true",
        help="write JSONL records to stdout instead of a file",
    )
    parser.add_argument(
        "--stream", action="store_true",
        help="keep producing fresh records until the consumer closes stdout",
    )
    args = parser.parse_args()
    if args.samples < 1 and not args.stream:
        raise ValueError("samples must be positive")
    config = StateVectorConfig()
    rng = random.Random(args.seed)
    if args.stdout or args.stream:
        index = 0
        try:
            while args.stream or index < args.samples:
                print(json.dumps(build_record(index, config, rng)), flush=True)
                index += 1
        except BrokenPipeError:
            # The consumer intentionally closes the pipe after its step budget.
            # Rebind stdout so interpreter shutdown does not flush the broken fd.
            sys.stdout = open(os.devnull, "w", encoding="utf-8")
            return
        return
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as stream:
        for index in range(args.samples):
            stream.write(json.dumps(build_record(index, config, rng)) + "\n")
    print(f"wrote {args.samples} v6 records to {args.out}")


if __name__ == "__main__":
    main()
