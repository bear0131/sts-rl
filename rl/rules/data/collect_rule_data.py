"""Collect single-combat state/action/effect/next-state rule samples.

At every main-trajectory decision, all legal complete simulator actions are
executed on independent BattleContext clones. One randomly selected legal action
then advances the main combat. A completed combat is discarded and an entirely
new loadout is sampled; this is intentionally not a combat gauntlet.
"""

from __future__ import annotations

import argparse
from collections import Counter
import gzip
import json
from pathlib import Path

import numpy as np

from env.core import _BUILD  # noqa: F401
from env.core import factories
from env.core.rule_language import (
    RULE_LANGUAGE_VERSION,
    deck_rule_sentence,
    relic_loadout_sentence,
    validate_first_version_language,
)
import slaythespire as sts
from schemas.combat_state import _sim_card_id, normalize_sim, normalize_sim_global

from rl.rules.rule_data import (
    RULE_DATA_VERSION,
    derive_local_calculations,
    derive_rule_steps,
    events_from_snapshots,
)


PLAYER_STATUS_RANGES = {
    "STRENGTH": (1, 30),
    "DEXTERITY": (1, 15),
    "WEAK": (1, 5),
    "FRAIL": (1, 5),
    "VULNERABLE": (1, 5),
    "ARTIFACT": (1, 5),
    "DOUBLE_DAMAGE": (1, 2),
    "INTANGIBLE": (1, 3),
    "BARRICADE": (1, 1),
    "CORRUPTION": (1, 1),
    "DOUBLE_TAP": (1, 2),
    "BRUTALITY": (1, 2),
    "COMBUST": (1, 7),
    "DARK_EMBRACE": (1, 2),
    "DEMON_FORM": (1, 5),
    "EVOLVE": (1, 3),
    "FEEL_NO_PAIN": (1, 8),
    "FIRE_BREATHING": (1, 12),
    "FLAME_BARRIER": (1, 15),
    "JUGGERNAUT": (1, 12),
    "METALLICIZE": (1, 15),
    "RAGE": (1, 12),
    "RUPTURE": (1, 3),
    "THORNS": (1, 15),
    "VIGOR": (1, 40),
}

MONSTER_STATUS_RANGES = {
    "STRENGTH": (1, 30),
    "VULNERABLE": (1, 5),
    "WEAK": (1, 5),
    "ARTIFACT": (1, 5),
    "METALLICIZE": (1, 15),
    "PLATED_ARMOR": (1, 15),
}


def _enum_tail(value) -> str:
    return str(value).split(".")[-1]


def _state(raw: dict, gc) -> dict:
    combat = normalize_sim(raw)
    global_state = normalize_sim_global(gc)
    # GameContext is intentionally not mutated during a standalone combat. Keep
    # the live combat-owned resources authoritative in the collected condition.
    global_state["hp"] = combat["player"]["hp"]
    global_state["max_hp"] = combat["player"]["max_hp"]
    global_state["gold"] = combat["player"].get("gold", global_state["gold"])
    return {"combat": combat, "global": global_state}


def _action_record(action, before: dict, bc) -> dict:
    kind = _enum_tail(action.action_type)
    record = {
        "type": kind,
        "source": int(action.source_idx),
        "target": int(action.target_idx),
    }
    if kind == "CARD":
        source = int(action.source_idx)
        if 0 <= source < len(before["combat"]["hand"]):
            record["card"] = dict(before["combat"]["hand"][source])
    elif kind == "POTION":
        source = int(action.source_idx)
        potions = list(bc.potions)
        if 0 <= source < len(potions):
            record["potion"] = potions[source]
    elif kind == "MULTI_CARD_SELECT":
        record["selected"] = list(action.selected_indices)
    if kind in ("SINGLE_CARD_SELECT", "MULTI_CARD_SELECT"):
        record["select_context"] = str(bc.select_info().get("context", "PAD"))
    return record


def _choose_main_action(actions, rng: np.random.Generator):
    card_actions = [a for a in actions if _enum_tail(a.action_type) == "CARD"]
    # Prefer card play so the random trajectory does not end every turn early,
    # while preserving a substantial chance of turn and select transitions.
    candidates = card_actions if card_actions and rng.random() < 0.75 else actions
    return candidates[int(rng.integers(0, len(candidates)))]


def _sample_for_action(
    gc, bc, action, encounter: str, combat_index: int, decision: int,
    *, shard_id: int, local_combat_index: int, injected: dict,
):
    before_raw = bc.to_dict()
    before = _state(before_raw, gc)
    action_record = _action_record(action, before, bc)
    branch = bc.clone()
    trace = branch.execute_with_trace(action)
    raw_snapshots = [trace["before"], *trace["steps"]]
    snapshots = [normalize_sim(raw) for raw in raw_snapshots]
    effects = events_from_snapshots(snapshots)
    after = _state(trace["after"], gc)
    sample = {
        "data_version": RULE_DATA_VERSION,
        "language_version": RULE_LANGUAGE_VERSION,
        "metadata": {
            "encounter": encounter,
            "combat_index": combat_index,
            "local_combat_index": local_combat_index,
            "shard_id": shard_id,
            "decision": decision,
            "act": int(gc.act),
            "floor": int(gc.floor_num),
            "injected": injected,
        },
        "state": before,
        "action": action_record,
        "rules": {
            "deck": deck_rule_sentence(before["global"]["deck"]),
            "relics": relic_loadout_sentence(before["global"]["relics"]),
        },
        "effects": effects,
        "next_state": after,
    }
    # Materialize the reasoning annotations in the dataset so they can be
    # inspected and independently validated before tokenization.
    sample["rule_steps"] = derive_rule_steps(sample)
    sample["local_calculations"] = derive_local_calculations(sample)
    return sample


def _sample_statuses(
    rng: np.random.Generator, ranges: dict[str, tuple[int, int]], max_count: int
) -> dict[str, int]:
    count = int(rng.integers(0, min(max_count, len(ranges)) + 1))
    if not count:
        return {}
    names = list(ranges)
    selected = rng.choice(len(names), size=count, replace=False)
    return {
        names[int(index)]: int(rng.integers(
            ranges[names[int(index)]][0], ranges[names[int(index)]][1] + 1
        ))
        for index in selected
    }


def _random_block(rng: np.random.Generator, maximum: int) -> int:
    if maximum <= 0 or rng.random() < 0.5:
        return 0
    return int(rng.integers(1, maximum + 1))


def _inject_random_combat_state(bc, rng: np.random.Generator, args) -> dict:
    if not args.random_statuses:
        return {"player_block": 0, "player_statuses": {}, "monsters": []}

    player_block = _random_block(rng, args.player_block_max)
    bc.set_player_block(player_block)
    player_statuses = _sample_statuses(
        rng, PLAYER_STATUS_RANGES, args.max_player_statuses
    )
    for name, amount in player_statuses.items():
        bc.set_player_status(name, amount)

    monsters = []
    raw_monsters = list(bc.to_dict()["monsters"])
    for target, monster in enumerate(raw_monsters):
        if bool(monster.get("gone", False)):
            monsters.append({"block": 0, "statuses": {}})
            continue
        block = _random_block(rng, args.monster_block_max)
        bc.set_monster_block(target, block)
        statuses = _sample_statuses(
            rng, MONSTER_STATUS_RANGES, args.max_monster_statuses
        )
        for name, amount in statuses.items():
            bc.set_monster_status(target, name, amount)
        monsters.append({"block": block, "statuses": statuses})
    return {
        "player_block": player_block,
        "player_statuses": player_statuses,
        "monsters": monsters,
    }


def _open_output(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "wt", encoding="utf-8", compresslevel=3)
    return path.open("w", encoding="utf-8")


def _update_coverage(coverage: dict[str, Counter], sample: dict) -> None:
    metadata = sample["metadata"]
    coverage["encounters"][metadata["encounter"]] += 1
    coverage["action_types"][sample["action"]["type"]] += 1
    card = sample["action"].get("card")
    if card is not None:
        coverage["played_cards"][card["id"]] += 1


def _update_combat_coverage(
    coverage: dict[str, Counter], before: dict, injected: dict
) -> None:
    for relic in before["global"]["relics"]:
        coverage["relics"][relic] += 1
    for name in injected["player_statuses"]:
        coverage["player_statuses"][name] += 1
    coverage["player_block_nonzero"][str(injected["player_block"] > 0)] += 1
    for monster in injected["monsters"]:
        coverage["monster_block_nonzero"][str(monster["block"] > 0)] += 1
        for name in monster["statuses"]:
            coverage["monster_statuses"][name] += 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("rl/data/rules-v1.jsonl"))
    parser.add_argument("--states", type=int, default=10_000,
                        help="Number of main-trajectory decision states to enumerate.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--ascension", type=int, default=0)
    parser.add_argument("--floor-lo", type=int, default=1)
    parser.add_argument("--floor-hi", type=int, default=50)
    parser.add_argument("--max-decisions-per-combat", type=int, default=300)
    parser.add_argument("--max-actions-per-state", type=int, default=0,
                        help="0 enumerates every legal action; positive values subsample.")
    parser.add_argument(
        "--random-statuses", action=argparse.BooleanOptionalAction, default=True,
        help="Inject random player and monster combat statuses after battle setup.",
    )
    parser.add_argument("--max-player-statuses", type=int, default=4)
    parser.add_argument("--max-monster-statuses", type=int, default=3)
    parser.add_argument("--player-block-max", type=int, default=40)
    parser.add_argument("--monster-block-max", type=int, default=60)
    parser.add_argument("--progress-every", type=int, default=100)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.states < 1:
        raise ValueError("--states must be positive")
    if not 1 <= args.floor_lo <= args.floor_hi <= 50:
        raise ValueError("floor range must satisfy 1 <= lo <= hi <= 50")
    if args.max_actions_per_state < 0:
        raise ValueError("--max-actions-per-state cannot be negative")
    if args.shard_id < 0:
        raise ValueError("--shard-id cannot be negative")
    if not 0 <= args.max_player_statuses <= len(PLAYER_STATUS_RANGES):
        raise ValueError("--max-player-statuses is out of range")
    if not 0 <= args.max_monster_statuses <= len(MONSTER_STATUS_RANGES):
        raise ValueError("--max-monster-statuses is out of range")
    if args.player_block_max < 0 or args.monster_block_max < 0:
        raise ValueError("block maxima cannot be negative")
    if not hasattr(sts.BattleContext, "execute_with_trace"):
        raise RuntimeError(
            "slaythespire was built without rule tracing; configure the dedicated "
            "build with -DSTS_ENABLE_RULE_TRACE=ON"
        )
    if args.random_statuses and not all(
        hasattr(sts.BattleContext, name) for name in (
            "set_player_block", "set_player_status", "set_monster_status"
        )
    ):
        raise RuntimeError("slaythespire rule build lacks random-status setters")

    # Fail before collection if the loadout generator grows beyond the language
    # contract. This covers all Ironclad/status/curse cards and every sampled relic.
    sampled_card_enums = set(factories._IRONCLAD_CARDS)
    sampled_cards = {_sim_card_id(card_id) for card_id in sampled_card_enums}
    sampled_cards.update(("Burn", "Dazed", "Slimed", "Void", "Wound"))
    sampled_relics = (
        set(factories._RELIC_POOL)
        | set(factories._BOSS_RELIC_POOL)
        | {"Burning Blood"}
    )
    validate_first_version_language(sampled_cards, sampled_relics)

    rng = np.random.default_rng(args.seed)
    make_combat = factories.random_rule_combat_factory(
        rng,
        ascension=args.ascension,
        floor_range=(args.floor_lo, args.floor_hi),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    states = samples = combats = 0
    coverage: dict[str, Counter] = {
        name: Counter() for name in (
            "encounters", "action_types", "played_cards", "relics",
            "player_statuses", "monster_statuses", "player_block_nonzero",
            "monster_block_nonzero",
        )
    }
    with _open_output(args.out) as stream:
        while states < args.states:
            gc, encounter = make_combat()
            bc = sts.create_battle(gc, encounter)
            injected = _inject_random_combat_state(bc, rng, args)
            combats += 1
            initial = _state(bc.to_dict(), gc)
            _update_combat_coverage(coverage, initial, injected)
            for decision in range(args.max_decisions_per_combat):
                if bc.is_battle_over or states >= args.states:
                    break
                legal = list(bc.rule_actions())
                if not legal:
                    break
                enumerated = legal
                if args.max_actions_per_state and len(legal) > args.max_actions_per_state:
                    indices = rng.choice(
                        len(legal), size=args.max_actions_per_state, replace=False
                    )
                    enumerated = [legal[int(index)] for index in sorted(indices)]
                for action in enumerated:
                    sample = _sample_for_action(
                        gc, bc, action, encounter,
                        args.shard_id * 1_000_000_000 + combats - 1,
                        decision,
                        shard_id=args.shard_id,
                        local_combat_index=combats - 1,
                        injected=injected,
                    )
                    stream.write(json.dumps(sample, ensure_ascii=True, separators=(",", ":")))
                    stream.write("\n")
                    samples += 1
                    _update_coverage(coverage, sample)
                main_action = _choose_main_action(legal, rng)
                bc.execute(main_action)
                states += 1
                if args.progress_every and states % args.progress_every == 0:
                    print(
                        f"[rule-data] states={states}/{args.states} "
                        f"samples={samples} combats={combats}"
                    )
    print(
        f"[rule-data] wrote {samples} samples from {states} states and "
        f"{combats} combats to {args.out.resolve()}"
    )
    stats_path = args.out.with_suffix(args.out.suffix + ".stats.json")
    stats_path.write_text(
        json.dumps({
            "shard_id": args.shard_id,
            "seed": args.seed,
            "states": states,
            "samples": samples,
            "combats": combats,
            "coverage": {
                name: dict(sorted(counter.items()))
                for name, counter in coverage.items()
            },
        }, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"[rule-data] coverage statistics: {stats_path.resolve()}")


if __name__ == "__main__":
    main()
