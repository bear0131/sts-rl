"""Probe the latest rule checkpoint with a synthetic three-hit attack card.

This is deliberately a small, self-contained probe.  ``TripleSeven`` is not in
the training vocabulary, so its name becomes ``<UNK>`` while the rule words,
digits, Strength, and Thorns remain ordinary model inputs.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import torch

from rl.rule_data import (
    EFFECTS_BEGIN,
    EFFECTS_END,
    NEXT_STATE_BEGIN,
    NEXT_STATE_END,
    OUTPUT_BEGIN,
    OUTPUT_END,
    RuleTokenizer,
    _card_tokens,
    _number,
    _sentence_tokens,
    serialize_calculations,
    serialize_effects,
    serialize_state,
)
from rl.rule_model import RuleWorldModel


RUN = Path("rl/runs/rules-random-v2-large256-dist4")
CARD = {"id": "TripleSeven", "up": 0, "cost": 1}
CARD_RULE = (
    "CARD TripleSeven VERSION base PROGRAM CARD_BEGIN TYPE_ATTACK "
    "RARITY_COMMON COST NUM_BEGIN DIGIT_1 NUM_END DO DEAL_DAMAGE "
    "NUM_BEGIN DIGIT_7 NUM_END TARGET_ENEMY TIMES NUM_BEGIN DIGIT_3 "
    "NUM_END STMT_END CARD_END"
)
DECK_RULE = "DECK ; COUNT 1 " + CARD_RULE


def _base_state(*, player_hp: int, player_thorns: int, monster_hp: int,
                monster_thorns: int) -> dict:
    counters = {
        "centennial_puzzle_active": False, "happy_flower": 0,
        "incense_burner": 0, "ink_bottle": 0, "inserter": 0,
        "lizard_tail_active": False, "nunchaku": 0, "pen_nib": 0,
        "sundial": 0, "necronomicon_used": False, "cards_played": 0,
        "attacks_played": 0, "skills_played": 0, "orange_attack": False,
        "orange_skill": False, "orange_power": False, "combust_hp_loss": 0,
        "deva_form_energy_per_turn": 0, "echo_form_cards_doubled": 0,
        "panache_counter": 0, "cards_discarded": 0, "times_damaged": 0,
        "bomb_1": 0, "bomb_2": 0, "bomb_3": 0,
        "last_attack_unblocked_damage": 0,
    }
    player_powers = {"Strength": 4}
    if player_thorns:
        player_powers["Thorns"] = player_thorns
    monster_powers = {}
    if monster_thorns:
        monster_powers["Thorns"] = monster_thorns
    return {
        "combat": {
            "turn": 1,
            "player": {
                "hp": player_hp, "max_hp": 80, "block": 0, "energy": 3,
                "gold": 99, "powers": player_powers,
                "combat_counters": counters,
            },
            "monsters": [{
                "id": "Cultist", "hp": monster_hp, "max_hp": 100,
                "block": 0, "gone": False, "base_damage": 6,
                "adjusted_damage": 6, "hits": 1, "powers": monster_powers,
                "move": "CULTIST_INCANTATION", "move_id": 50,
            }],
            "hand": [dict(CARD)], "draw": [], "discard": [], "exhaust": [],
        },
        "global": {
            "character": "IRONCLAD", "ascension": 0, "act": 1, "floor": 1,
            "hp": player_hp, "max_hp": 80, "boss": "HEXAGHOST",
            "relics": [], "relic_counters": [],
            "potions": ["EMPTY_POTION_SLOT"] * 3, "gold": 99,
            "deck": [["TripleSeven", 0]], "boss_relics": [],
        },
    }


def _action_prefix(state: dict) -> list[str]:
    out = serialize_state(state, include_rules=False)
    out.extend(("RULE_CONTEXT_BEGIN", "DECK_RULE_BEGIN"))
    out.extend(_sentence_tokens(DECK_RULE))
    out.extend(("DECK_RULE_END", "RELIC_RULE_BEGIN"))
    out.extend(_sentence_tokens("RELIC_LOADOUT"))
    out.extend(("RELIC_RULE_END", "RULE_CONTEXT_END"))
    out.extend(("ACTION_BEGIN", "ACTION_CARD", "ACTION_FIELD_SOURCE"))
    out.extend(_number(0))
    out.extend(("ACTION_FIELD_TARGET",))
    out.extend(_number(0))
    out.extend(_card_tokens(CARD))
    out.extend(("ACTION_RULE_BEGIN", *_sentence_tokens(CARD_RULE),
                "ACTION_RULE_END", "ACTION_END"))
    return out


def _events(*, player_hp: int, monster_hp: int, player_thorns: int,
            monster_thorns: int) -> list[dict]:
    deltas = []
    current_player, current_monster = player_hp, monster_hp
    atomic = 0
    deltas.extend((
        {"type": "PLAYER_VALUE_CHANGE", "atomic": 0, "field": "energy",
         "before": 3, "delta": -1, "after": 2},
        {"type": "COUNTER_CHANGE", "atomic": 0, "counter": "attacks_played",
         "before": 0, "delta": 1, "after": 1},
        {"type": "COUNTER_CHANGE", "atomic": 0, "counter": "cards_played",
         "before": 0, "delta": 1, "after": 1},
        {"type": "CARD_REMOVE", "atomic": 0, "source": "hand",
         "card": "TripleSeven", "up": 0},
    ))
    for _ in range(3):
        before_monster = current_monster
        current_monster -= 11
        deltas.append({
            "type": "MONSTER_VALUE_CHANGE", "atomic": atomic + 1,
            "slot": 0, "field": "hp", "before": before_monster,
            "delta": -11, "after": current_monster,
        })
        if monster_thorns:
            before_player = current_player
            current_player -= monster_thorns
            deltas.append({
                "type": "PLAYER_VALUE_CHANGE", "atomic": atomic + 2,
                "field": "hp", "before": before_player,
                "delta": -monster_thorns, "after": current_player,
            })
        atomic += 2 if monster_thorns else 1
    deltas.extend((
        {"type": "CARD_ADD", "atomic": atomic + 1, "destination": "discard",
         "card": "TripleSeven", "up": 0},
        {"type": "ACTION_COMPLETE", "atomic_steps": atomic + 2},
    ))
    return deltas


def _sample(*, enemy_thorns: int, player_thorns: int) -> dict:
    hp = 80
    monster_hp = 100
    before = _base_state(
        player_hp=hp, player_thorns=player_thorns,
        monster_hp=monster_hp, monster_thorns=enemy_thorns,
    )
    after = copy.deepcopy(before)
    player = after["combat"]["player"]
    monster = after["combat"]["monsters"][0]
    player["hp"] = 80 - enemy_thorns * 3
    player["energy"] = 2
    player["combat_counters"]["cards_played"] = 1
    player["combat_counters"]["attacks_played"] = 1
    monster["hp"] = 67
    after["combat"]["hand"] = []
    after["combat"]["discard"] = [dict(CARD, cost=None)]
    after["combat"]["discard"][0].pop("cost", None)
    after["global"]["hp"] = player["hp"]
    effects = _events(
        player_hp=80, monster_hp=100, player_thorns=player_thorns,
        monster_thorns=enemy_thorns,
    )
    return {
        "name": "enemy_thorns" if enemy_thorns else "player_thorns",
        "state": before, "next_state": after, "effects": effects,
    }


def _target(sample: dict) -> list[str]:
    return [
        OUTPUT_BEGIN,
        *serialize_calculations([]),
        *serialize_effects(
            sample["effects"], least_significant_first=True,
            canonical_card_tokens=False,
        ),
        NEXT_STATE_BEGIN,
        *serialize_state(sample["next_state"], include_rules=False,
                         least_significant_first=True),
        NEXT_STATE_END, OUTPUT_END,
    ]


def _pad(rows: list[list[int]], pad_id: int) -> torch.Tensor:
    return torch.nn.utils.rnn.pad_sequence(
        [torch.tensor(row, dtype=torch.long) for row in rows],
        batch_first=True, padding_value=pad_id,
    )


def _read_number(tokens: list[str], pos: int):
    if pos >= len(tokens) or tokens[pos] not in ("NUM_BEGIN", "NUM_LSD_BEGIN"):
        return None, pos
    lsd = tokens[pos] == "NUM_LSD_BEGIN"
    pos += 1
    sign = -1 if pos < len(tokens) and tokens[pos] == "SIGN_NEG" else 1
    if sign < 0:
        pos += 1
    digits = []
    while pos < len(tokens) and tokens[pos].startswith("DIGIT_"):
        digits.append(tokens[pos][-1])
        pos += 1
    if not digits or pos >= len(tokens) or tokens[pos] != "NUM_END":
        return None, pos
    pos += 1
    text = "".join(reversed(digits) if lsd else digits)
    return sign * int(text), pos


def _predicted_values(tokens: list[str]) -> dict:
    effects = []
    try:
        end = tokens.index(EFFECTS_END)
    except ValueError:
        end = len(tokens)
    pos = 0
    while pos < end:
        try:
            begin = tokens.index("EFFECT_BEGIN", pos, end)
            stop = tokens.index("EFFECT_END", begin, end)
        except ValueError:
            break
        chunk = tokens[begin:stop]
        event_type = next((x.removeprefix("EFFECT_") for x in chunk
                           if x.startswith("EFFECT_")
                           and x not in ("EFFECT_BEGIN", "EFFECT_END")), "")
        event = {"type": event_type}
        for field in ("field", "before", "delta", "after"):
            marker = f"FIELD_{field}"
            if marker in chunk:
                at = chunk.index(marker) + 1
                if field == "field":
                    event[field] = chunk[at].removeprefix("VALUE_") if at < len(chunk) else None
                else:
                    event[field], _ = _read_number(chunk, at)
        effects.append(event)
        pos = stop + 1
    return {
        "monster_hp_deltas": [e.get("delta") for e in effects
                               if e.get("type") == "MONSTER_VALUE_CHANGE"
                               and e.get("field") == "hp"],
        "player_hp_deltas": [e.get("delta") for e in effects
                              if e.get("type") == "PLAYER_VALUE_CHANGE"
                              and e.get("field") == "hp"],
        "effects": effects,
    }


def main() -> None:
    torch.set_num_threads(8)
    tokenizer = RuleTokenizer.load(RUN / "tokenizer.json")
    model = RuleWorldModel.from_config(RUN / "model_config.json")
    checkpoint = torch.load(RUN / "checkpoint-epoch32-test.pt",
                            map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    samples = [_sample(enemy_thorns=2, player_thorns=0),
               _sample(enemy_thorns=0, player_thorns=3)]
    encoded = []
    for sample in samples:
        prefix = _action_prefix(sample["state"])
        target = _target(sample)
        encoded.append((prefix, target))
        print(json.dumps({
            "case": sample["name"], "prefix_tokens": len(prefix),
            "target_tokens": len(target),
            "unknown_prefix": sum(x == "<UNK>" for x in tokenizer.decode(tokenizer.encode(prefix))),
            "unknown_target": sum(x == "<UNK>" for x in tokenizer.decode(tokenizer.encode(target))),
        }, ensure_ascii=False))

    with torch.no_grad():
        prefixes = _pad([tokenizer.encode(x[0]) for x in encoded], tokenizer.pad_id)
        targets = _pad([tokenizer.encode(x[1]) for x in encoded], tokenizer.pad_id)
        logits, _ = model(prefixes, targets[:, :-1])
        for row, sample in enumerate(samples):
            expected = encoded[row][1]
            ids = logits[row, :len(expected) - 1].argmax(dim=-1).tolist()
            teacher = [expected[0]] + tokenizer.decode(ids)
            values = _predicted_values(teacher)
            print(json.dumps({"case": sample["name"], "mode": "teacher_forced",
                              **values}, ensure_ascii=False))

        generated = model.generate(
            prefixes, output_begin_id=tokenizer.index[OUTPUT_BEGIN],
            output_end_id=tokenizer.index[OUTPUT_END], max_new_tokens=900,
        ).tolist()
    for row, sample in enumerate(samples):
        tokens = tokenizer.decode(generated[row])
        values = _predicted_values(tokens)
        print(json.dumps({"case": sample["name"], "mode": "autoregressive",
                          "terminated": OUTPUT_END in tokens,
                          "generated_tokens": len(tokens), **values}, ensure_ascii=False))
        Path(f"rl/runs/rules-random-v2-large256-dist4/triple_seven_{sample['name']}.json").write_text(
            json.dumps({"tokens": tokens, **values}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
