"""Collect a controlled Strike/Bash data set with random player Strength."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from env.core import _BUILD  # noqa: F401
import slaythespire as sts

from rl.collect_rule_data import _enum_tail, _sample_for_action
from rl.rule_data import controlled_attack_calculations, derive_local_calculations


DECK = [("STRIKE_RED", 0)] * 5 + [("BASH", 0)] * 5
CARD_ENUMS = ("STRIKE_RED", "BASH")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=4_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--strength-min", type=int, default=0)
    parser.add_argument("--strength-max", type=int, default=9)
    parser.add_argument(
        "--strength-values",
        help="comma-separated explicit Strength values; overrides min/max",
    )
    parser.add_argument("--enemy-block-max", type=int, default=0)
    parser.add_argument("--enemy-vulnerable-max", type=int, default=0)
    parser.add_argument("--encounter", default="Cultist")
    parser.add_argument("--progress-every", type=int, default=500)
    return parser


def _card_actions(bc, wanted_card: str):
    raw_hand = bc.to_dict()["hand"]
    return [
        action for action in bc.rule_actions()
        if _enum_tail(action.action_type) == "CARD"
        and raw_hand[int(action.source_idx)][0] == wanted_card
    ]


def _damage_from_effects(effects) -> int:
    return sum(
        -int(effect["delta"])
        for effect in effects
        if effect["type"] == "MONSTER_VALUE_CHANGE"
        and effect["field"] == "hp"
        and int(effect["delta"]) < 0
    )


def _strength_pool(args) -> tuple[int, ...]:
    if args.strength_values is None:
        if not 0 <= args.strength_min <= args.strength_max:
            raise ValueError("strength range must satisfy 0 <= min <= max")
        return tuple(range(args.strength_min, args.strength_max + 1))
    try:
        values = tuple(sorted({
            int(piece.strip())
            for piece in args.strength_values.split(",")
            if piece.strip()
        }))
    except ValueError as exc:
        raise ValueError("--strength-values must contain comma-separated integers") from exc
    if not values or values[0] < 0:
        raise ValueError("--strength-values must contain non-negative integers")
    return values


def main() -> None:
    args = build_parser().parse_args()
    if args.samples < 1:
        raise ValueError("--samples must be positive")
    strength_pool = _strength_pool(args)
    if args.enemy_block_max < 0 or args.enemy_vulnerable_max < 0:
        raise ValueError("enemy status maxima must be non-negative")
    if not hasattr(sts.BattleContext, "set_player_strength"):
        raise RuntimeError(
            "slaythespire was built without controlled rule experiments; "
            "configure with -DSTS_ENABLE_RULE_TRACE=ON"
        )
    if args.enemy_block_max and not hasattr(sts.BattleContext, "set_monster_block"):
        raise RuntimeError("slaythespire lacks controlled monster Block support")
    if args.enemy_vulnerable_max and not hasattr(
        sts.BattleContext, "set_monster_vulnerable"
    ):
        raise RuntimeError("slaythespire lacks controlled monster Vulnerable support")

    rng = np.random.default_rng(args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    counts = {"Strike_R": 0, "Bash": 0}
    strengths = {value: 0 for value in strength_pool}
    with args.out.open("w", encoding="utf-8") as stream:
        for sample_index in range(args.samples):
            wanted = CARD_ENUMS[sample_index % len(CARD_ENUMS)]
            status_case = (sample_index // len(CARD_ENUMS)) % 4
            strength = int(strength_pool[int(rng.integers(0, len(strength_pool)))])
            enemy_block = (
                int(rng.integers(1, args.enemy_block_max + 1))
                if args.enemy_block_max and status_case in (1, 3)
                else 0
            )
            enemy_vulnerable = (
                int(rng.integers(1, args.enemy_vulnerable_max + 1))
                if args.enemy_vulnerable_max and status_case in (2, 3)
                else 0
            )
            for _attempt in range(100):
                game_seed = int(rng.integers(0, 2**31 - 1))
                gc = sts.GameContext(sts.CharacterClass.IRONCLAD, game_seed, 0)
                gc.floor_num = 1
                gc.set_deck(DECK)
                gc.set_relics([])
                bc = sts.create_battle(gc, args.encounter)
                bc.set_player_strength(strength)
                if args.enemy_block_max:
                    bc.set_monster_block(0, enemy_block)
                if args.enemy_vulnerable_max:
                    bc.set_monster_vulnerable(0, enemy_vulnerable)
                actions = _card_actions(bc, wanted)
                if actions:
                    break
            else:
                raise RuntimeError(f"could not draw {wanted} in 100 fresh combats")

            action = actions[int(rng.integers(0, len(actions)))]
            sample = _sample_for_action(
                gc, bc, action, args.encounter, sample_index, 0
            )
            card_id = sample["action"]["card"]["id"]
            before_monster = sample["state"]["combat"]["monsters"][0]
            after_monster = sample["next_state"]["combat"]["monsters"][0]
            hp_loss = int(before_monster["hp"]) - int(after_monster["hp"])
            block_loss = int(before_monster["block"]) - int(after_monster["block"])
            base_damage = 6 if card_id == "Strike_R" else 8
            calculated_damage = base_damage + strength
            if enemy_vulnerable:
                calculated_damage = calculated_damage * 3 // 2
            if hp_loss + block_loss != calculated_damage:
                raise AssertionError(
                    "controlled attack damage does not match Strength/Vulnerable/Block formula"
                )
            next_vulnerable = int(
                after_monster.get("powers", {}).get("Vulnerable", 0)
            )
            sample["metadata"].update({
                "experiment": "random_strength_enemy_status_strike_bash",
                "strength": strength,
                "card": card_id,
                "enemy_block": enemy_block,
                "enemy_vulnerable": enemy_vulnerable,
                "calculated_attack_damage": calculated_damage,
                "expected_block_loss": block_loss,
                "expected_hp_loss": hp_loss,
                "expected_damage": hp_loss,
                "expected_next_block": int(after_monster["block"]),
                "expected_next_hp": int(after_monster["hp"]),
                "expected_next_vulnerable": next_vulnerable,
            })
            sample["calculations"] = controlled_attack_calculations(sample)
            sample["local_calculations"] = derive_local_calculations(sample)
            del sample["calculations"]
            if sample["state"]["global"]["relics"]:
                raise AssertionError("controlled sample unexpectedly has relics")
            if {card for card, _up in sample["state"]["global"]["deck"]} != {
                "Strike_R", "Bash"
            }:
                raise AssertionError("controlled sample contains an unexpected card")
            stream.write(json.dumps(sample, ensure_ascii=True, separators=(",", ":")))
            stream.write("\n")
            counts[card_id] += 1
            strengths[strength] += 1
            if args.progress_every and (sample_index + 1) % args.progress_every == 0:
                print(f"[strength-data] {sample_index + 1}/{args.samples}")

    print(
        f"[strength-data] wrote={args.samples} cards={counts} "
        f"strengths={strengths} out={args.out.resolve()}"
    )


if __name__ == "__main__":
    main()
