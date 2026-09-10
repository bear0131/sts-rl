"""Train and probe an isolated name-to-register-value retrieval task.

The environment-owned spans create registers and present query names.  The
model owns only the result marker and a fixed-width two-digit value for each
query.  Samples are generated on demand so no example is reused.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import random
from typing import Iterable, Sequence

import torch
from torch import nn
from torch.nn import functional as F


PAD = "<PAD>"
SPECIAL = (
    "<REGISTERS_BEGIN>", "<REGISTERS_END>", "REG_CREATE", "REG_VALUE_BEGIN",
    "REG_VALUE_END", "<QUERIES_BEGIN>", "<QUERIES_END>", "REG_QUERY",
    "<QUERY_BEGIN>", "<QUERY_RESPONSE_BEGIN>", "<QUERY_RESPONSE_END>",
    "<QUERY_END>", "REG_RESULT", "NUM_LSD_BEGIN", "NUM_END",
)
REGISTER_NAMES = tuple(f"REG_NAME_{index}" for index in range(8))
DIGITS = tuple(f"DIGIT_{digit}" for digit in range(10))
TOKENS = (PAD, *SPECIAL, *REGISTER_NAMES, *DIGITS)
TOKEN_TO_ID = {token: index for index, token in enumerate(TOKENS)}
ID_TO_TOKEN = tuple(TOKENS)
PAD_ID = TOKEN_TO_ID[PAD]


def _number_tokens(value: int) -> tuple[str, ...]:
    value = int(value) % 100
    return ("NUM_LSD_BEGIN", f"DIGIT_{value % 10}",
            f"DIGIT_{value // 10}", "NUM_END")


@dataclass(frozen=True)
class RegisterSample:
    values: tuple[int, ...]
    queries: tuple[int, ...]
    tokens: tuple[str, ...]
    model_mask: tuple[bool, ...]
    result_spans: tuple[tuple[int, int], ...]

    @property
    def ids(self) -> tuple[int, ...]:
        return tuple(TOKEN_TO_ID[token] for token in self.tokens)


def make_sample(
    rng: random.Random,
    *,
    register_count: int | None = None,
    query_count: int | None = None,
    repeated_queries: int = 4,
    values: Sequence[int] | None = None,
    queries: Sequence[int] | None = None,
    definition_order: Sequence[int] | None = None,
    balanced_queries: bool = False,
) -> RegisterSample:
    count = register_count or rng.randint(4, 8)
    if values is None:
        # Unique values make a wrong-register copy distinguishable from a
        # correct name lookup during the mutation and swap probes.
        values = tuple(rng.sample(range(100), count))
    else:
        values = tuple(int(value) % 100 for value in values)
        count = len(values)
    if count < 2 or count > len(REGISTER_NAMES):
        raise ValueError("register count must be between 2 and 8")
    if queries is None:
        total = query_count or rng.randint(12, 20)
        if balanced_queries:
            query_list = [index for index in range(count)
                          for _ in range(max(1, total // count))]
            query_list.extend(
                rng.randrange(count) for _ in range(total - len(query_list))
            )
        else:
            target = rng.randrange(count)
            query_list = [target] * min(repeated_queries, total)
            query_list.extend(rng.randrange(count) for _ in range(total - len(query_list)))
        rng.shuffle(query_list)
        queries = tuple(query_list)
    else:
        queries = tuple(int(index) for index in queries)
    if not queries or any(index < 0 or index >= count for index in queries):
        raise ValueError("query index is outside the register table")

    definitions = list(range(count)) if definition_order is None else list(definition_order)
    if definition_order is None:
        rng.shuffle(definitions)
    if sorted(definitions) != list(range(count)):
        raise ValueError("definition_order must contain each register exactly once")
    tokens: list[str] = ["<REGISTERS_BEGIN>"]
    model_mask: list[bool] = [False]
    for index in definitions:
        tokens.extend(("REG_CREATE", REGISTER_NAMES[index], "REG_VALUE_BEGIN"))
        tokens.extend(_number_tokens(values[index]))
        tokens.append("REG_VALUE_END")
        model_mask.extend(False for _ in range(3 + 4 + 1))
    tokens.append("<REGISTERS_END>")
    model_mask.append(False)
    tokens.append("<QUERIES_BEGIN>")
    model_mask.append(False)
    spans: list[tuple[int, int]] = []
    for index in queries:
        tokens.extend((
            "<QUERY_BEGIN>", "REG_QUERY", REGISTER_NAMES[index],
            "<QUERY_RESPONSE_BEGIN>", "REG_RESULT", *_number_tokens(values[index]),
        ))
        start = len(tokens) - 5
        end = len(tokens)
        spans.append((start, end))
        tokens.extend(("<QUERY_RESPONSE_END>", "<QUERY_END>"))
        # Query boundaries and the response header are environment-owned; only
        # the two digits test retrieval from the register table.
        model_mask.extend((
            False, False, False, False, False, False, True, True, False,
            False, False,
        ))
    tokens.append("<QUERIES_END>")
    model_mask.append(False)
    if len(tokens) != len(model_mask):
        raise RuntimeError("sample token and ownership lengths diverged")
    return RegisterSample(
        tuple(values), tuple(queries), tuple(tokens), tuple(model_mask), tuple(spans),
    )


class RegisterTransformer(nn.Module):
    def __init__(self, d_model: int = 128, n_heads: int = 4,
                 n_layers: int = 3, feedforward: int = 256,
                 max_tokens: int = 256) -> None:
        super().__init__()
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.embedding = nn.Embedding(len(TOKENS), d_model, padding_idx=PAD_ID)
        self.position = nn.Embedding(max_tokens, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=feedforward,
            dropout=0.0, activation="gelu", batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer, num_layers=n_layers, enable_nested_tensor=False,
        )
        self.norm = nn.LayerNorm(d_model)
        self.output = nn.Linear(d_model, len(TOKENS), bias=False)
        self.output.weight = self.embedding.weight
        self.max_tokens = max_tokens

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        _, length = input_ids.shape
        if length > self.max_tokens:
            raise ValueError("sequence exceeds model capacity")
        hidden = self.embedding(input_ids)
        hidden = hidden + self.position(torch.arange(length, device=input_ids.device))[None]
        causal = torch.triu(
            torch.ones(length, length, dtype=torch.bool, device=input_ids.device),
            diagonal=1,
        )
        padding = input_ids.eq(PAD_ID)
        hidden = self.transformer(hidden, mask=causal, src_key_padding_mask=padding)
        return self.output(self.norm(hidden))


def _batch(
    samples: Sequence[RegisterSample], device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    width = max(len(sample.tokens) for sample in samples)
    input_ids = torch.full((len(samples), width - 1), PAD_ID, dtype=torch.long)
    labels = torch.full_like(input_ids, PAD_ID)
    owned = torch.zeros_like(input_ids, dtype=torch.bool)
    for row, sample in enumerate(samples):
        ids = torch.as_tensor(sample.ids, dtype=torch.long)
        length = len(ids) - 1
        input_ids[row, :length] = ids[:-1]
        labels[row, :length] = ids[1:]
        owned[row, :length] = torch.as_tensor(sample.model_mask[1:], dtype=torch.bool)
    return input_ids.to(device), labels.to(device), owned.to(device)


def _value_from_result(tokens: Sequence[str], span: tuple[int, int]) -> int | None:
    start, end = span
    body = tokens[start:end]
    if len(body) != 5 or body[0] != "REG_RESULT":
        return None
    if body[1] != "NUM_LSD_BEGIN" or body[4] != "NUM_END":
        return None
    if not body[2].startswith("DIGIT_") or not body[3].startswith("DIGIT_"):
        return None
    return int(body[2][6:]) + 10 * int(body[3][6:])


def _ar_decode(
    model: RegisterTransformer, sample: RegisterSample, device: torch.device,
) -> tuple[list[str], list[int]]:
    expected = list(sample.ids)
    generated = torch.as_tensor([[expected[0]]], dtype=torch.long, device=device)
    output_ids = [expected[0]]
    model.eval()
    with torch.inference_mode():
        for position in range(1, len(expected)):
            logits = model(generated)[:, -1]
            if sample.model_mask[position]:
                next_id = int(logits.argmax(dim=-1).item())
            else:
                next_id = expected[position]
            output_ids.append(next_id)
            generated = torch.cat(
                (generated, torch.as_tensor([[next_id]], device=device)), dim=1,
            )
    return [ID_TO_TOKEN[index] for index in output_ids], output_ids


def _ar_decode_batch(
    model: RegisterTransformer, samples: Sequence[RegisterSample],
    device: torch.device,
) -> list[list[str]]:
    """Decode equal-length samples together; environment positions are injected."""

    if not samples:
        return []
    lengths = {len(sample.tokens) for sample in samples}
    if len(lengths) != 1:
        raise ValueError("batched register decoding requires equal lengths")
    expected = torch.as_tensor([sample.ids for sample in samples], dtype=torch.long, device=device)
    generated = expected[:, :1]
    output = [list(row) for row in expected[:, :1].tolist()]
    model.eval()
    with torch.inference_mode():
        for position in range(1, expected.shape[1]):
            logits = model(generated)[:, -1]
            next_ids = expected[:, position].clone()
            for row, sample in enumerate(samples):
                if sample.model_mask[position]:
                    next_ids[row] = int(logits[row].argmax().item())
            output = [
                output[row] + [int(next_ids[row])]
                for row in range(len(samples))
            ]
            generated = torch.cat((generated, next_ids[:, None]), dim=1)
    return [[ID_TO_TOKEN[index] for index in row] for row in output]


def _teacher_metrics(
    model: RegisterTransformer, samples: Sequence[RegisterSample], device: torch.device,
) -> tuple[int, int]:
    input_ids, labels, owned = _batch(samples, device)
    with torch.inference_mode():
        predictions = model(input_ids).argmax(dim=-1)
    correct = int(((predictions == labels) & owned).sum().item())
    total = int(owned.sum().item())
    return correct, total


def _evaluate(
    model: RegisterTransformer, samples: Iterable[RegisterSample], device: torch.device,
) -> dict[str, float]:
    samples = list(samples)
    teacher_correct = teacher_total = query_correct = query_total = 0
    span_correct = span_total = consistent = 0
    for start in range(0, len(samples), 128):
        chunk = samples[start:start + 128]
        correct, total = _teacher_metrics(model, chunk, device)
        teacher_correct += correct
        teacher_total += total
        for length_start in range(0, len(chunk), 128):
            group = chunk[length_start:length_start + 128]
            by_length: dict[int, list[RegisterSample]] = {}
            for sample in group:
                by_length.setdefault(len(sample.tokens), []).append(sample)
            for same_length in by_length.values():
                predictions = _ar_decode_batch(model, same_length, device)
                for sample, predicted in zip(same_length, predictions):
                    predicted_values = [
                        _value_from_result(predicted, span) for span in sample.result_spans
                    ]
                    expected_values = [sample.values[index] for index in sample.queries]
                    query_total += len(expected_values)
                    query_correct += sum(
                        value == expected
                        for value, expected in zip(predicted_values, expected_values)
                    )
                    span_total += len(expected_values)
                    span_correct += sum(
                        predicted[span[0]:span[1]] == list(sample.tokens[span[0]:span[1]])
                        for span in sample.result_spans
                    )
                    for register in set(sample.queries):
                        seen = [
                            predicted_values[i] for i, index in enumerate(sample.queries)
                            if index == register
                        ]
                        if len(seen) > 1 and len(set(seen)) == 1:
                            consistent += 1
    return {
        "teacher_token_accuracy": teacher_correct / max(teacher_total, 1),
        "teacher_tokens": float(teacher_total),
        "autoregressive_query_accuracy": query_correct / max(query_total, 1),
        "autoregressive_queries": float(query_total),
        "autoregressive_span_accuracy": span_correct / max(span_total, 1),
        "repeated_register_consistency": consistent,
    }


def _mutation_probe(
    model: RegisterTransformer, rng: random.Random, device: torch.device,
) -> dict[str, object]:
    base = make_sample(
        rng, register_count=4, query_count=12, repeated_queries=6,
        balanced_queries=True,
    )
    definition_order = [
        int(token.removeprefix("REG_NAME_"))
        for token in base.tokens
        if token.startswith("REG_NAME_")
    ][:len(base.values)]
    target = max(set(base.queries), key=base.queries.count)
    changed_values = list(base.values)
    changed_values[target] = (changed_values[target] + 37) % 100
    changed = make_sample(
        rng, values=changed_values, queries=base.queries,
        definition_order=definition_order,
    )
    swapped_values = list(base.values)
    other = next(index for index in range(len(swapped_values)) if index != target)
    swapped_values[target], swapped_values[other] = swapped_values[other], swapped_values[target]
    swapped = make_sample(
        rng, values=swapped_values, queries=base.queries,
        definition_order=definition_order,
    )
    base_pred, _ = _ar_decode(model, base, device)
    changed_pred, _ = _ar_decode(model, changed, device)
    swapped_pred, _ = _ar_decode(model, swapped, device)

    def values(sample: RegisterSample, prediction: Sequence[str]) -> list[int | None]:
        return [_value_from_result(prediction, span) for span in sample.result_spans]

    base_values = values(base, base_pred)
    changed_values_pred = values(changed, changed_pred)
    swapped_values_pred = values(swapped, swapped_pred)
    target_positions = [i for i, index in enumerate(base.queries) if index == target]
    other_positions = [i for i, index in enumerate(base.queries) if index != target]
    target_follow = sum(
        changed_values_pred[i] == changed_values[target] for i in target_positions
    )
    other_unchanged = sum(
        changed_values_pred[i] == base_values[i] for i in other_positions
    )
    swap_expected = [swapped_values[index] for index in base.queries]
    swap_correct = sum(
        swapped_values_pred[i] == swap_expected[i] for i in range(len(swap_expected))
    )
    return {
        "register_values": list(base.values),
        "query_registers": list(base.queries),
        "target_register": target,
        "changed_value": changed_values[target],
        "baseline_predictions": base_values,
        "changed_predictions": changed_values_pred,
        "target_queries_followed_mutation": target_follow,
        "target_query_count": len(target_positions),
        "other_queries_unchanged": other_unchanged,
        "other_query_count": len(other_positions),
        "name_swap_values": swapped_values,
        "name_swap_predictions": swapped_values_pred,
        "name_swap_correct": swap_correct,
        "name_swap_query_count": len(swap_expected),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--eval-samples", type=int, default=64)
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", type=str, default="docs/register-retrieval-experiment.json")
    # Match the intended standalone probe capacity: a four-layer causal
    # Transformer is enough to represent key/value lookup without making the
    # experiment depend on the much larger full world model.
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--feedforward", type=int, default=256)
    return parser


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def main() -> None:
    args = build_parser().parse_args()
    if args.steps < 1 or args.batch_size < 1 or args.eval_samples < 1:
        raise ValueError("steps, batch-size and eval-samples must be positive")
    device = _device(args.device)
    train_rng = random.Random(args.seed)
    val_rng = random.Random(args.seed + 1)
    probe_rng = random.Random(args.seed + 2)
    torch.set_float32_matmul_precision("high")
    model = RegisterTransformer(
        d_model=args.d_model,
        n_heads=args.heads,
        n_layers=args.layers,
        feedforward=args.feedforward,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)
    model.train()
    history: list[dict[str, float]] = []
    for step in range(1, args.steps + 1):
        samples = [make_sample(
            train_rng, register_count=4, query_count=12, balanced_queries=True,
        ) for _ in range(args.batch_size)]
        inputs, labels, owned = _batch(samples, device)
        logits = model(inputs)
        loss_values = F.cross_entropy(logits[owned], labels[owned])
        optimizer.zero_grad(set_to_none=True)
        loss_values.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % args.eval_every == 0 or step == args.steps:
            val_samples = [make_sample(
                val_rng, register_count=4, query_count=12, balanced_queries=True,
            ) for _ in range(args.eval_samples)]
            metrics = _evaluate(model, val_samples, device)
            row = {
                "step": float(step), "loss": float(loss_values.detach()), **metrics,
            }
            history.append(row)
            print(
                f"step={step} loss={row['loss']:.4f} "
                f"tf_acc={row['teacher_token_accuracy']:.4f} "
                f"ar_query_acc={row['autoregressive_query_accuracy']:.4f}",
                flush=True,
            )
    mutation = _mutation_probe(model, probe_rng, device)
    report = {
        "task": "isolated_register_name_retrieval",
        "device": str(device),
        "seed": args.seed,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "data_generation": "fresh random samples per optimization step",
        "protocol": {
            "register_definition": "environment-owned name plus fixed two-digit LSD value",
            "query": "environment-owned REG_QUERY and register name",
            "model_owned": "REG_RESULT plus NUM_LSD_BEGIN, two DIGIT tokens and NUM_END",
            "repeated_queries": "each register is queried three times in random order",
        },
        "model": {
            "d_model": args.d_model,
            "heads": args.heads,
            "layers": args.layers,
            "feedforward": args.feedforward,
            "vocab_size": len(TOKENS),
        },
        "history": history,
        "mutation_probe": mutation,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"final": history[-1], "mutation_probe": mutation}, ensure_ascii=False), flush=True)
    print(f"report={output}")


if __name__ == "__main__":
    main()
