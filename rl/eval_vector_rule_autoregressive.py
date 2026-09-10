"""Evaluate a trained v6 vector-rule checkpoint autoregressively.

The evaluator deliberately builds the two documented examples from anonymous
rule programs.  Card and relic names are used only by this environment-side
adapter to retrieve their rule programs; the model receives rule-token vectors
and fixed state vectors.  The report includes both token-level agreement and
the continuous delta error at predicted ``DELTA_SLOT`` positions.

Example::

    python -m rl.eval_vector_rule_autoregressive \
        --checkpoint .tmp-current-vector-v6-last.pt \
        --device cpu --max-new-tokens 256 \
        --output docs/vector-v6-autoregressive-eval.json
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from env.core.card_language import TOKEN_INDEX as CARD_RULE_INDEX, card_program
from rl.render_rule_vector_examples import (
    _fly_kick,
    _pommel_with_sundial,
    _state,
    render_serialized,
)
from rl.rule_vector_model import (
    CARD_RULE_VOCAB,
    RELIC_RULE_IDS,
    RELIC_RULE_VOCAB,
    SerializedVectorEvents,
    StateVectorConfig,
    VECTOR_TOKEN_INDEX,
    VectorRuleModelConfig,
    VectorRuleWorldModel,
    vectorize_state,
)


@dataclass(frozen=True)
class EvaluationCase:
    name: str
    state: Mapping[str, Any]
    action_card: str
    action_upgrade: int
    hand_cards: tuple[tuple[str, int], ...]
    relic_ids: tuple[str, ...]
    expected: SerializedVectorEvents


def build_evaluation_cases() -> tuple[EvaluationCase, ...]:
    """Return the two fixed examples used by the v6 documentation."""

    fly_state = _state(
        energy=3,
        cards_played=0,
        attacks_played=0,
        sundial=0,
        enemy_hp=50,
        enemy_block=0,
        draw=5,
        discard=1,
        enemy_powers={"Vulnerable": 2},
    )
    pommel_state = _state(
        energy=3,
        cards_played=0,
        attacks_played=0,
        sundial=2,
        enemy_hp=60,
        enemy_block=0,
        draw=0,
        discard=3,
        relics=("Sundial",),
        relic_counters=(2,),
    )
    return (
        EvaluationCase(
            name="fly-kick-vulnerable",
            state=fly_state,
            action_card="Dropkick",
            action_upgrade=0,
            hand_cards=(("Dropkick", 0), ("Strike_R", 0), ("Defend_R", 0)),
            relic_ids=(),
            expected=_fly_kick(),
        ),
        EvaluationCase(
            name="pommel-strike-shuffle-sundial",
            state=pommel_state,
            action_card="Pommel Strike",
            action_upgrade=0,
            hand_cards=(("Pommel Strike", 0), ("Strike_R", 0), ("Defend_R", 0)),
            relic_ids=("Sundial",),
            expected=_pommel_with_sundial(),
        ),
    )


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _card_program(card_id: str, upgraded: int) -> tuple[int, ...]:
    return tuple(
        CARD_RULE_INDEX[token]
        for token in card_program(card_id, bool(upgraded))
    )


def _padded_programs(
    programs: Sequence[Sequence[int]], *, pad_id: int, device: torch.device
) -> torch.Tensor:
    width = max((len(program) for program in programs), default=1)
    result = torch.full(
        (len(programs), width), pad_id, dtype=torch.long, device=device
    )
    for row, program in enumerate(programs):
        if program:
            result[row, :len(program)] = torch.as_tensor(
                program, dtype=torch.long, device=device
            )
    return result


def _model_inputs(
    model: VectorRuleWorldModel,
    case: EvaluationCase,
    config: VectorRuleModelConfig,
    device: torch.device,
) -> tuple[dict[str, Any], list[str]]:
    action_program = _padded_programs(
        [_card_program(case.action_card, case.action_upgrade)],
        pad_id=model.card_encoder.config.pad_id,
        device=device,
    )
    action_vector = model.card_encoder.forward_programs(action_program)

    hand_programs = [
        _card_program(card_id, upgrade)
        for card_id, upgrade in case.hand_cards
    ]
    hand_program_ids = _padded_programs(
        hand_programs,
        pad_id=model.card_encoder.config.pad_id,
        device=device,
    )
    hand_vectors = model.card_encoder.forward_programs(hand_program_ids).unsqueeze(0)
    hand_upgrades = torch.as_tensor(
        [[float(upgrade) for _card_id, upgrade in case.hand_cards]],
        dtype=torch.float32,
        device=device,
    )
    hand_mask = torch.ones(
        (1, len(case.hand_cards)), dtype=torch.bool, device=device
    )

    relic_vectors = None
    relic_mask = None
    unsupported_relics: list[str] = []
    if case.relic_ids:
        supported_relics = tuple(
            relic_id for relic_id in case.relic_ids if relic_id in RELIC_RULE_IDS
        )
        unsupported_relics = [
            relic_id for relic_id in case.relic_ids if relic_id not in RELIC_RULE_IDS
        ]
        if supported_relics:
            relic_vectors = model.relic_encoder.vectors_for_ids(
                supported_relics, device=device
            ).unsqueeze(0)
            relic_mask = torch.ones(
                (1, len(supported_relics)), dtype=torch.bool, device=device
            )

    state = vectorize_state(case.state, config.state, device=device)
    return ({
        "state": state,
        "action_card_vector": action_vector,
        "action_upgrade": torch.as_tensor(
            [float(case.action_upgrade)], dtype=torch.float32, device=device
        ),
        "hand_card_vectors": hand_vectors,
        "hand_upgrades": hand_upgrades,
        "hand_mask": hand_mask,
        "relic_vectors": relic_vectors,
        "relic_mask": relic_mask,
    }, unsupported_relics)


def _token_names(ids: Sequence[int]) -> tuple[str, ...]:
    inverse = {index: token for token, index in VECTOR_TOKEN_INDEX.items()}
    return tuple(inverse.get(int(index), "<UNK>") for index in ids)


def _predicted_serialized(
    token_names: Sequence[str], deltas: torch.Tensor
) -> SerializedVectorEvents:
    """Align model.generate's one-step delta rows with generated tokens."""

    targets: list[tuple[float, ...] | None] = []
    for position, token in enumerate(token_names):
        if token == "DELTA_SLOT" and position > 0 and position - 1 < deltas.shape[0]:
            targets.append(tuple(float(value) for value in deltas[position - 1].tolist()))
        else:
            targets.append(None)
    return SerializedVectorEvents(tuple(token_names), tuple(targets))


def _token_metrics(
    expected: Sequence[str], predicted: Sequence[str]
) -> dict[str, Any]:
    common = min(len(expected), len(predicted))
    matching = sum(expected[index] == predicted[index] for index in range(common))
    first_mismatch = None
    for index in range(common):
        if expected[index] != predicted[index]:
            first_mismatch = index
            break
    if first_mismatch is None and len(expected) != len(predicted):
        first_mismatch = common
    return {
        "exact_match": list(expected) == list(predicted),
        "expected_tokens": len(expected),
        "predicted_tokens": len(predicted),
        "matching_prefix_tokens": matching,
        "token_accuracy_over_max_length": matching / max(len(expected), len(predicted), 1),
        "first_mismatch": first_mismatch,
        "expected_at_mismatch": (
            expected[first_mismatch] if first_mismatch is not None and first_mismatch < len(expected) else None
        ),
        "predicted_at_mismatch": (
            predicted[first_mismatch] if first_mismatch is not None and first_mismatch < len(predicted) else None
        ),
    }


def _delta_metrics(
    expected: SerializedVectorEvents,
    predicted: SerializedVectorEvents,
) -> dict[str, Any]:
    expected_rows = [
        delta for token, delta in zip(expected.tokens, expected.delta_targets)
        if token == "DELTA_SLOT" and delta is not None
    ]
    predicted_rows = [
        delta for token, delta in zip(predicted.tokens, predicted.delta_targets)
        if token == "DELTA_SLOT" and delta is not None
    ]
    count = min(len(expected_rows), len(predicted_rows))
    if not count:
        return {
            "expected_delta_slots": len(expected_rows),
            "predicted_delta_slots": len(predicted_rows),
            "compared_delta_slots": 0,
            "mean_absolute_error": None,
        }
    absolute_error = 0.0
    values = 0
    for left, right in zip(expected_rows[:count], predicted_rows[:count]):
        if len(left) != len(right):
            raise ValueError("expected and predicted delta widths differ")
        absolute_error += sum(abs(float(a) - float(b)) for a, b in zip(left, right))
        values += len(left)
    return {
        "expected_delta_slots": len(expected_rows),
        "predicted_delta_slots": len(predicted_rows),
        "compared_delta_slots": count,
        "mean_absolute_error": absolute_error / max(values, 1),
    }


def evaluate_case(
    model: VectorRuleWorldModel,
    case: EvaluationCase,
    config: VectorRuleModelConfig,
    device: torch.device,
    max_new_tokens: int,
) -> dict[str, Any]:
    inputs, unsupported_relics = _model_inputs(model, case, config, device)
    output_begin_id = VECTOR_TOKEN_INDEX["<OUTPUT_BEGIN>"]
    output_end_id = VECTOR_TOKEN_INDEX["<OUTPUT_END>"]
    with torch.no_grad():
        generated, deltas = model.generate(
            **inputs,
            output_begin_id=output_begin_id,
            output_end_id=output_end_id,
            max_new_tokens=max_new_tokens,
        )
    predicted_tokens = _token_names(generated[0].tolist())
    predicted = _predicted_serialized(predicted_tokens, deltas[0])
    token_metrics = _token_metrics(case.expected.tokens, predicted.tokens)
    delta_metrics = _delta_metrics(case.expected, predicted)
    return {
        "name": case.name,
        "action_card": case.action_card,
        "relic_ids": list(case.relic_ids),
        "unsupported_relics": unsupported_relics,
        "token_metrics": token_metrics,
        "delta_metrics": delta_metrics,
        "expected_tokens": list(case.expected.tokens),
        "predicted_tokens": list(predicted.tokens),
        "expected_rendered": render_serialized(case.expected, config.state),
        # Early checkpoints often emit tiny noise in every delta dimension.
        # Keep the raw vectors for metrics, but render only material changes.
        "predicted_rendered": render_serialized(
            predicted, config.state, min_delta_abs=0.005
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", type=Path,
        default=Path(".tmp-current-vector-v6-last.pt"),
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--output", type=Path)
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.max_new_tokens < 1:
        raise ValueError("--max-new-tokens must be positive")
    device = _device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config_payload = checkpoint.get("config")
    if not isinstance(config_payload, Mapping):
        raise ValueError("checkpoint does not contain model config")
    config = VectorRuleModelConfig.from_dict(config_payload)
    if config.card_encoder.token_vocab_size != len(CARD_RULE_VOCAB):
        raise ValueError("checkpoint card vocabulary does not match this checkout")
    if config.relic_encoder.token_vocab_size != len(RELIC_RULE_VOCAB):
        raise ValueError("checkpoint relic vocabulary does not match this checkout")
    model = VectorRuleWorldModel(config).to(device)
    model.load_state_dict(checkpoint["model"])
    max_new_tokens = min(args.max_new_tokens, config.decoder.max_tokens - 1)
    results = [
        evaluate_case(model, case, config, device, max_new_tokens)
        for case in build_evaluation_cases()
    ]
    report = {
        "checkpoint": str(args.checkpoint),
        "device": str(device),
        "max_new_tokens": max_new_tokens,
        "checkpoint_step": checkpoint.get("step"),
        "cases": results,
    }
    return report


def _print_report(report: Mapping[str, Any]) -> None:
    print(
        f"checkpoint={report['checkpoint']} device={report['device']} "
        f"max_new_tokens={report['max_new_tokens']}"
    )
    for case in report["cases"]:
        token = case["token_metrics"]
        delta = case["delta_metrics"]
        print(
            f"\n=== {case['name']} ===\n"
            f"exact_match={token['exact_match']} "
            f"prefix={token['matching_prefix_tokens']}/{token['expected_tokens']} "
            f"predicted_tokens={token['predicted_tokens']} "
            f"first_mismatch={token['first_mismatch']}\n"
            f"delta_slots={delta['compared_delta_slots']}/"
            f"{delta['expected_delta_slots']} "
            f"mae={delta['mean_absolute_error']}"
        )
        print("-- expected --")
        print(case["expected_rendered"])
        print("-- predicted --")
        print(case["predicted_rendered"])


def main() -> None:
    args = build_parser().parse_args()
    report = run(args)
    _print_report(report)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"report={args.output}")


if __name__ == "__main__":
    main()
