"""Inspect one SELF_DAMAGE local decode without running the full case suite.

This diagnostic intentionally uses the current producer unchanged.  It is
useful for separating a model's numeric prediction error from a bad producer
label at lethal HP.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import torch

from rl.evaluation.rules.eval_vector_rule_v7_cases import (
    EvaluationCase,
    _describe_case,
    _fixed_v6_record,
    _render_tokens,
    evaluate_case,
)
from rl.rules.data.make_vector_rule_v7_data import compile_v7_record
from rl.rules.rule_vector_model import (
    StateVectorConfig,
    VectorRuleModelConfig,
    VectorRuleWorldModel,
)
from rl.rules.vector_rule_v7_data import V7TokenCodec, V7_EXECUTOR_CONTRACT_VERSION


def _device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def _state(player_hp: int) -> dict[str, Any]:
    return {
        "combat": {
            "turn": 1,
            "player": {
                "hp": int(player_hp),
                "max_hp": 80,
                "block": 0,
                "energy": 3,
                "gold": 99,
                "powers": {},
                "combat_counters": {
                    "cards_played": 0,
                    "attacks_played": 0,
                    "sundial": 0,
                },
            },
            "monsters": [{
                "id": "ExampleEnemy",
                "hp": 60,
                "max_hp": 60,
                "block": 0,
                "base_damage": 0,
                "adjusted_damage": 0,
                "hits": 0,
                "gone": False,
                "powers": {},
            }],
            "hand": [],
            "draw": [],
            "discard": [],
            "exhaust": [],
        },
        "global": {
            "floor": 1,
            "act": 1,
            "relics": [],
            "relic_counters": [],
            "deck": [],
        },
    }


def _find_self_damage(node: Mapping[str, Any]) -> Mapping[str, Any] | None:
    if node.get("kind") == "SELF_DAMAGE":
        return node
    for child in node.get("children", ()):
        found = _find_self_damage(child)
        if found is not None:
            return found
    return None


def build_case(config: StateVectorConfig, player_hp: int) -> EvaluationCase:
    hand = (("Bloodletting", 0),)
    canonical = _state(player_hp)
    v6_record, canonical = _fixed_v6_record(
        index=-10,
        state=canonical,
        action_id="Bloodletting",
        action_upgrade=0,
        hand=hand,
        config=config,
        seed=0,
    )
    return EvaluationCase(
        name="self-damage-bloodletting",
        group="diagnostic",
        canonical=canonical,
        record=compile_v7_record(v6_record, config),
        action_id="Bloodletting",
        action_upgrade=0,
        hand=hand,
        relics=(),
        note="single fixed SELF_DAMAGE diagnostic",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--player-hp", type=int, default=10)
    parser.add_argument("--device", default="auto")
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.player_hp < 0:
        raise ValueError("player-hp must be non-negative")
    device = _device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if int(checkpoint.get("executor_contract_version", -1)) != V7_EXECUTOR_CONTRACT_VERSION:
        raise ValueError("checkpoint uses an incompatible executor contract")
    model_config = VectorRuleModelConfig.from_dict(checkpoint["config"])
    codec = V7TokenCodec.load(args.tokenizer)
    if model_config.decoder.token_vocab_size != len(codec.tokens):
        raise ValueError("checkpoint decoder vocabulary does not match tokenizer")
    model = VectorRuleWorldModel(model_config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    case = build_case(model_config.state, args.player_hp)
    result = evaluate_case(
        model,
        case,
        codec,
        model_config,
        device,
        min(512, model_config.decoder.max_tokens - 1),
    )
    self_node = _find_self_damage(case.record["effect_tree"])
    self_local = next(
        row for row in result["local"] if row["kind"] == "SELF_DAMAGE"
    )
    report = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": int(checkpoint.get("step", 0)),
        "device": str(device),
        "player_hp_before": int(args.player_hp),
        "card": "Bloodletting",
        "case_input": _describe_case(case),
        "self_damage_node": self_node,
        "self_damage_local_decode": self_local,
        "current_producer_formula": "max(1, player_hp_before - trigger_damage)",
        "comparison_formula_without_hp_lock": "max(0, player_hp_before - trigger_damage)",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "output": str(args.output),
        "checkpoint_step": report["checkpoint_step"],
        "player_hp_before": report["player_hp_before"],
        "self_damage_metrics": self_local["metrics"],
        "self_damage_node": self_node,
    }, ensure_ascii=False, indent=2))
    print("\nEXPECTED LOCAL TOKENS\n" + _render_tokens(self_local["expected_tokens"]))
    print("\nPREDICTED LOCAL TOKENS\n" + _render_tokens(self_local["predicted_tokens"]))
    return report


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
