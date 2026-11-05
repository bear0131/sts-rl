"""Online shared-chain distillation for a two-layer entity policy and its native guide."""

import numpy as np
import torch

from rl.policies.entity_action_playout import ActionTokenPolicy, action_tokens

TOKEN_REGRET_CONTRACT = "sts1-paired-regret-root64x2-guide-v1"
GUIDE_WEIGHT = 0.5


def encode_public_roots(raw):
    """Preserve individual legal-action source and target handles, including extra selection tokens."""
    tables = [
        {
            k: np.concatenate((x["entities"][k], x["inventory"][k]))
            for k in x["entities"]
        }
        for x in raw
    ]
    lengths = np.array([len(x["ids"]) for x in tables])
    candidates = np.array([len(x["action_bits"]) for x in raw])
    data = {k: np.concatenate([x[k] for x in tables]) for k in tables[0]}
    data.update(
        lengths=lengths,
        alens=candidates,
        sources=np.concatenate([x["source_rows"] for x in raw]),
        targets=np.concatenate([x["target_slots"] for x in raw]),
    )
    fixed, symbols, extra, extra_ids, counts, sources, targets = action_tokens(data)
    result = []
    start = action_start = 0
    for i, (n, a) in enumerate(zip(counts, candidates, strict=True)):
        f = fixed[i]
        ids = symbols[i]
        if n:
            f = np.concatenate((f, extra[start : start + n]))
            ids = np.concatenate((ids, extra_ids[start : start + n]))
        sl = slice(action_start, action_start + a)
        result.append((f, ids, sources[sl], targets[sl]))
        start += n
        action_start += a
    return result


def pack_rows(rows, device):
    n = len(rows)
    sizes = np.array([len(row["action_bits"]) for row in rows], np.int64)
    width = int(sizes.max())
    ends = np.cumsum(sizes)
    within = np.arange(ends[-1]) - np.repeat(ends - sizes, sizes)
    slots = np.repeat(np.arange(n), sizes) * width + within
    mask = np.arange(width)[None, :] < sizes[:, None]
    tokens = [row["tokens"] for row in rows]
    lengths = [len(t[0]) for t in tokens]
    max_tokens = max(lengths)
    if min(lengths) == max_tokens:
        features = np.stack([t[0] for t in tokens])
        symbols = np.stack([t[1] for t in tokens])
    else:
        features = np.zeros((n, max_tokens, tokens[0][0].shape[1]), np.float32)
        symbols = np.zeros((n, max_tokens), np.int32)
        for i, (f, s, _, _) in enumerate(tokens):
            features[i, : len(f)] = f
            symbols[i, : len(s)] = s
    arrays = (
        np.stack([row["state"] for row in rows]),
        np.concatenate([row["actions"] for row in rows]),
        np.concatenate([row["keys"] for row in rows]).astype(np.int64),
        slots,
        features,
        symbols.astype(np.int64),
        np.concatenate([t[2] for t in tokens]).astype(np.int64),
        np.concatenate([t[3] for t in tokens]).astype(np.int64),
        mask,
    )
    return tuple(torch.as_tensor(a, device=device) for a in arrays)


def improved_log_target(row):
    visits = np.asarray(row["visits"])
    raw = np.asarray(row["prior_logits"], np.float64)
    values = np.asarray(row["values"], np.float64)
    observed = visits > 0
    if (
        visits.shape != raw.shape
        or raw.shape != values.shape
        or not observed.any()
        or not np.isfinite(values[observed]).all()
        or not np.isfinite(raw).all()
        or (visits < 0).any()
        or not np.isfinite(row["return_units"])
        or row["return_units"] <= 0
    ):
        raise ValueError("invalid measured combat action returns")
    logp = np.full(len(values), -np.inf)
    logp[observed] = raw[observed] + np.log(2.0) * row["return_units"] * (
        values[observed] - values[observed].max()
    )
    logp[observed] -= np.logaddexp.reduce(logp[observed])
    return logp.astype(np.float32)


class OnlineRegretSamples:
    def __init__(self, rows):
        self.rows = rows
        self.log_targets = [improved_log_target(row) for row in rows]
        self.teacher = np.array([row.argmax() for row in self.log_targets])
        flattened = np.concatenate(self.log_targets).astype(np.float64)
        finite = flattened[np.isfinite(flattened)]
        self.target_entropy = float(-(np.exp(finite) * finite).sum())
        self.uniform_ce = sum(np.log(len(row)) for row in self.log_targets)

    def batch(self, indices, device):
        rows = [self.rows[i] for i in indices]
        inputs = pack_rows(rows, device)
        logp = np.full(tuple(inputs[-1].shape), -np.inf, np.float32)
        for j, i in enumerate(indices):
            t = self.log_targets[i]
            logp[j, : len(t)] = t
        return (
            inputs,
            torch.as_tensor(logp, device=device),
            torch.as_tensor(self.teacher[indices], device=device),
        )


class OnlineTokenRegretPolicy(ActionTokenPolicy):
    def __init__(self, key_count):
        super().__init__(key_count, coupled=True, width=64, depth=2)

    def export_guide(self):
        from rl.policies.small_playout import SmallPlayoutPolicy
        return SmallPlayoutPolicy.export(self)

    def collection_pair(self, *, collect_samples):
        from env.search.small_playout import PAIRED_CONTRACT
        from rl.algorithms.upper_probe import state_digest
        from rl.policies.small_playout import export_online_search_pair

        pair = export_online_search_pair(self, collect_samples=collect_samples, guide_only=True)
        pair.update(search_contract=PAIRED_CONTRACT, sha256=state_digest(self.state_dict()))
        return pair

    def prepare_samples(self, samples, target_mix):
        return OnlineRegretSamples(samples)

    def update_terms(self, inputs, log_target):
        full, guide = self.scores_and_guide(*inputs)
        observed = torch.isfinite(log_target)
        lp = (
            full.masked_fill(~observed, -torch.inf)
            .log_softmax(-1)
            .masked_fill(~observed, 0.0)
        )
        target = log_target.masked_fill(~observed, 0.0)
        p = lp.exp().masked_fill(~observed, 0.0)
        # KL(student || improved prior) has exactly the same gradient as expected
        # action regret + KL(student || frozen collection policy).
        reverse = (p * (lp - target)).sum()
        forward = (log_target.exp() * (target - lp)).sum()
        legal = inputs[-1]
        root_lp = full.log_softmax(-1).masked_fill(~legal, 0.0)
        teacher = root_lp.detach()
        guide_lp = guide.log_softmax(-1).masked_fill(~legal, 0.0)
        guide_kl = (teacher.exp().masked_fill(~legal, 0.0) * (teacher - guide_lp)).sum()
        entropy = -(root_lp.exp() * root_lp * legal).sum()
        return reverse + GUIDE_WEIGHT * guide_kl, entropy, root_lp, forward, guide_kl
