"""Run one schema-v7 optimizer step at a forced decoder padding width."""

from __future__ import annotations

import argparse
import random

import torch
import torch.nn.functional as F

from rl.rules.data.make_vector_rule_v7_data import build_v7_record
from rl.rules.rule_vector_model import (
    CARD_RULE_VOCAB,
    RELIC_RULE_VOCAB,
    VectorRuleModelConfig,
    VectorRuleWorldModel,
)
from rl.rules.training.train_rule_vectors_v7 import _loss, _move_batch
from rl.rules.vector_rule_v7_data import (
    V7TokenCodec,
    collate_vector_rule_v7_batch,
    examples_from_v7_record,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--pad-to", type=int, default=186)
    parser.add_argument("--seed", type=int, default=12901)
    parser.add_argument("--device", default="cuda")
    return parser


def _pad_decoder(batch: dict, width: int, pad_id: int) -> None:
    current = int(batch["target_input_ids"].shape[1])
    if current > width:
        raise ValueError(f"natural decoder width {current} exceeds requested {width}")
    amount = width - current
    if amount == 0:
        return
    batch["target_input_ids"] = F.pad(
        batch["target_input_ids"], (0, amount), value=pad_id,
    )
    batch["labels"] = F.pad(batch["labels"], (0, amount), value=pad_id)
    batch["delta_targets"] = F.pad(
        batch["delta_targets"], (0, 0, 0, amount), value=0.0,
    )
    batch["delta_mask"] = F.pad(
        batch["delta_mask"], (0, amount), value=False,
    )


def main() -> None:
    args = build_parser().parse_args()
    if args.batch_size < 1 or args.pad_to < 1:
        raise ValueError("batch-size and pad-to must be positive")
    device = torch.device(args.device)
    config = VectorRuleModelConfig.load(args.config)
    codec = V7TokenCodec.load(args.tokenizer)
    rng = random.Random(args.seed)
    examples = []
    record_index = 0
    while len(examples) < args.batch_size:
        record = build_v7_record(record_index, config.state, rng)
        examples.extend(examples_from_v7_record(
            record, codec, config.state,
            card_vocab_size=len(CARD_RULE_VOCAB),
            relic_vocab_size=len(RELIC_RULE_VOCAB),
            max_cards=config.hand_encoder.max_cards,
            max_relics=config.world_encoder.max_relics,
            max_tokens=config.decoder.max_tokens,
        ))
        record_index += 1
    batch = collate_vector_rule_v7_batch(
        examples[:args.batch_size], pad_id=codec.pad_id,
    )
    natural_width = int(batch["target_input_ids"].shape[1])
    _pad_decoder(batch, args.pad_to, codec.pad_id)
    batch = _move_batch(batch, device)

    model = VectorRuleWorldModel(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    optimizer.zero_grad(set_to_none=True)
    loss, token_loss, delta_loss, _output = _loss(model, batch, 1.0)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    peak_allocated = (
        torch.cuda.max_memory_allocated(device) / (1024 ** 3)
        if device.type == "cuda" else 0.0
    )
    peak_reserved = (
        torch.cuda.max_memory_reserved(device) / (1024 ** 3)
        if device.type == "cuda" else 0.0
    )
    print(
        f"batch_size={args.batch_size} natural_width={natural_width} "
        f"forced_width={args.pad_to} loss={float(loss.detach()):.6f} "
        f"token_loss={float(token_loss):.6f} delta_loss={float(delta_loss):.6f} "
        f"peak_allocated_gib={peak_allocated:.3f} "
        f"peak_reserved_gib={peak_reserved:.3f}"
    )


if __name__ == "__main__":
    main()
