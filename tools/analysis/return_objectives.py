"""Cost-sensitive and paired preference losses for frozen combat-policy comparisons."""

import numpy as np
import torch
import torch.nn.functional as F


def prepare_evidence(bank, evidence_paths, label_paths):
    lengths, pairs, compared = [], [], []
    for directory, labels in zip(evidence_paths, label_paths, strict=True):
        with (
            np.load(directory / "paired-evidence.npz") as data,
            np.load(labels / "labels.npz") as original,
        ):
            if not np.array_equal(data["ids"], original["ids"]):
                raise ValueError("pair evidence does not match source decisions")
            n = data["lengths"]
            v = data["pairs"]
            den = data["compared"]
            if len(n) != len(den) or v.shape != (n.sum(), 4) or (den < 1).any():
                raise ValueError("invalid preference bank")
            sizes = np.repeat(original["sizes"], n)
            if (
                (v[:, :2] < 0).any()
                or (v[:, :2] >= sizes[:, None]).any()
                or not np.isin(v[:, 2], [-1, 1]).all()
                or not np.isfinite(v).all()
                or (v[:, 3] <= 0).any()
            ):
                raise ValueError("invalid paired comparison")
            lengths.append(n)
            pairs.append(v)
            compared.append(den)
    bank.pair_lengths = np.concatenate(lengths)
    bank.pair_starts = bank.pair_lengths.cumsum() - bank.pair_lengths
    bank.pair_rows = np.concatenate(pairs)
    bank.pair_den = np.concatenate(compared)


def objective_loss(name, bank, part, packed_source, logits, legal, observed):
    """Return the sum of per-root losses, so existing microbatch averaging is unchanged."""
    logp = logits.log_softmax(-1).masked_fill(~observed, 0)
    prob = logp.exp().masked_fill(~observed, 0)
    old = torch.full_like(logp, -torch.inf)
    old[legal] = torch.as_tensor(
        bank.prior_logp[packed_source], dtype=logp.dtype, device=logp.device
    )
    old = (
        old.masked_fill(~observed, -torch.inf).log_softmax(-1).masked_fill(~observed, 0)
    )
    if name == "regret":
        cost = torch.zeros_like(logp)
        cost[legal] = torch.as_tensor(
            bank.action_cost[packed_source], dtype=cost.dtype, device=cost.device
        )
        return (prob * (np.log(2.0) * cost + logp - old)).sum()
    if name != "preferences":
        raise ValueError("unknown supervision objective")
    # A weak prior keeps decisions with no reliable paired difference close to the mature policy.
    loss = 0.25 * (old.exp().masked_fill(~observed, 0) * (old - logp)).sum()
    n = bank.pair_lengths[part]
    size = int(n.sum())
    if size == 0:
        return loss
    starts = n.cumsum() - n
    rows = bank.pair_rows[
        np.repeat(bank.pair_starts[part] - starts, n) + np.arange(size)
    ]
    owners = torch.as_tensor(np.repeat(np.arange(len(n)), n), device=logp.device)
    a, b = (
        torch.as_tensor(rows[:, i].astype(np.int64), device=logp.device) for i in (0, 1)
    )
    sign = torch.as_tensor(rows[:, 2], device=logp.device)
    weight = torch.as_tensor(
        (rows[:, 3] / np.repeat(bank.pair_den[part], n)).astype(np.float32),
        device=logp.device,
    )
    return (
        loss + (weight * F.softplus(-sign * (logp[owners, a] - logp[owners, b]))).sum()
    )
