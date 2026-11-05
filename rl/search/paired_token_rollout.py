"""Batch full entity-policy root scores after immutable native paired-chain searches."""

import numpy as np
import torch
import slaythespire as sts

from env.search.small_playout import SmallPlayoutBatch
from rl.policies.token_regret import encode_public_roots, pack_rows


class PairedTokenBatch(SmallPlayoutBatch):
    def __init__(self, envs, pool, pair, policy):
        super().__init__(envs, pool, pair)
        if not hasattr(sts.SmallPlayoutConfig(), "paired_worldlines"):
            raise ValueError("native shared-chain search is required")
        self.paired_worldlines = True
        self.policy = policy
        self.device = next(policy.parameters()).device

    @torch.inference_mode()
    def score_roots(self, environments):
        raw = [
            sts.small_playout_public_features(env.thick.gc, env.thick.bc)
            for env in environments
        ]
        tokens = encode_public_roots(raw)
        rows = [
            dict(
                state=x["state"],
                actions=x["actions"],
                keys=x["keys"],
                action_bits=x["action_bits"],
                tokens=t,
            )
            for x, t in zip(raw, tokens, strict=True)
        ]
        inputs = pack_rows(rows, self.device)
        scores = self.policy.forward_packed(*inputs).cpu().numpy()
        return [
            (row, scores[i, : len(row["action_bits"])].copy())
            for i, row in enumerate(rows)
        ]

    def decisions(self, calls, *, pre_step=None, start_steps=None):
        if pre_step is not None or start_steps is not None:
            raise ValueError("paired token roots require scoring before advancing their environment")
        out, jobs = self.prepare(calls)
        if jobs:
            results = sts.small_playout_search_batch(self.pool, self.net,
                [self.envs[i].thick.bc for i, _, _, _ in jobs],
                [cfg for _, _, cfg, _ in jobs], [nonce for _, _, _, nonce in jobs], self.prior)
            out.update(self.accept_ready(list(zip(jobs, results, strict=True))))
        return {i: out[i] for i in calls}

    def dome_policy(self, env, cfg):
        row, scores = self.score_roots([env])[0]
        return row["action_bits"], scores

    def accept_ready(self, completed):
        out = {}
        ranked = [(job, z) for job, z in completed if len(z["action_bits"]) > 1]
        by_slot = {}
        for start in range(0, len(ranked), 128):
            group = ranked[start : start + 128]
            sources = self.score_roots([self.envs[job[0]] for job, z in group])
            for (job, z), (row, logits) in zip(group, sources, strict=True):
                bits = np.asarray(z["action_bits"], np.uint32)
                if not np.array_equal(row["action_bits"], bits):
                    raise RuntimeError(
                        "paired search changed the public root action order"
                    )
                visits = np.asarray(z["visits"])
                q = np.asarray(z["values"], np.float64)
                eligible = np.flatnonzero(
                    (visits > 0) & (q >= q[visits > 0].max() - 0.005)
                )
                tied = eligible[logits[eligible] == logits[eligible].max()]
                rng = np.random.default_rng(job[3] ^ 0x524F4F54544945)
                z["action"] = int(bits[rng.choice(tied)])
                units = (
                    100.0 if job[2].boss else float(round(float(row["state"][1]) * 200))
                )
                by_slot[job[0]] = dict(
                    tokens=row["tokens"],
                    values=q.astype(np.float32),
                    prior_logits=logits,
                    return_units=units,
                )
        for job, z in completed:
            result = self.accept(job, z)
            if job[0] in by_slot:
                self.samples[-1].update(by_slot[job[0]])
            if result is not None:
                out[job[0]] = result
        return out
