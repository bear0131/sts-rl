"""Batch full entity-policy root scores after immutable native paired-chain searches."""

import numpy as np
import torch
import slaythespire as sts

from env.search.small_playout import SmallPlayoutBatch, PairedWorldPlayoutStream
from rl.policies.token_regret import encode_public_roots, pack_rows


class PairedTokenBatch(SmallPlayoutBatch):
    def __init__(self, envs, pool, pair, policy):
        super().__init__(envs, pool, pair)
        if not self.paired_worldlines or not hasattr(sts.SmallPlayoutConfig(), "paired_worldlines"):
            raise ValueError("native shared-chain search is required")
        self.policy = policy
        self.device = next(policy.parameters()).device
        if self.deck:
            from rl.search.rollout_inference import UpperPolicyInference
            self.inference = UpperPolicyInference(policy, self.device)
            self.inference.__enter__()
        if hasattr(sts, 'small_playout_world_batch'):
            self.stream_type = PairedWorldPlayoutStream

    @torch.inference_mode()
    def score_roots(self, environments, observations=None, contexts=None):
        if self.deck:
            inputs = (self.observer.encode(environments) if observations is None else
                      {key:np.stack([obs[key] for obs in observations]) for key in observations[0]})
            observations = [{key:value[i] for key,value in inputs.items()} for i in range(len(environments))]
            from rl.policies.deck_latent import SearchContext
            from rl.search.native import observation_as_tensor
            if contexts is None:contexts=[None]*len(environments)
            missing=[i for i,value in enumerate(contexts) if value is None]
            if missing:
                context=self.inference.search_context({k:v[missing] for k,v in inputs.items()}).combat.cpu().numpy()
                for i,value in zip(missing,context,strict=True):contexts[i]=value
                self.stats['root_context_cpu_rows']=self.stats.get('root_context_cpu_rows',0)+len(missing)
            projected=SearchContext(torch.as_tensor(np.stack(contexts),device=self.device),None)
            logits, _ = self.policy.combat_with_context(observation_as_tensor(inputs,self.device),projected)
            logits = logits.cpu().numpy()
            out = []
            for env, obs, scores in zip(environments, observations, logits, strict=True):
                mapping = sorted(env._joint_policy_actions())
                choices, bits = zip(*mapping)
                out.append((dict(observation=obs, action_bits=np.asarray(bits,np.uint32)),
                            scores[np.asarray(choices)].copy()))
            return out
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
            results = self.search_jobs(self.pool, jobs)
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
            sources = self.score_roots([self.envs[job[0]] for job, z in group],
                observations=[z['observation'] for _,z in group],
                contexts=[z.get('combat_context') for _,z in group]) if self.deck else self.score_roots([self.envs[job[0]] for job,z in group])
            for (job, z), (row, logits) in zip(group, sources, strict=True):
                bits = np.asarray(z["action_bits"], np.uint32)
                if not np.array_equal(row["action_bits"], bits):
                    raise RuntimeError(
                        "paired search changed the public root action order"
                    )
                visits = np.asarray(z["visits"])
                q = np.asarray(z["values"], np.float64)
                from env.search.action_selection import search_filtered_choice
                chosen, _ = search_filtered_choice(logits, q, visits, job[3])
                z["action"] = int(bits[chosen])
                units = (100.0 if job[2].boss else float(z['max_hp'])) if self.deck else (
                    100.0 if job[2].boss else float(round(float(row["state"][1]) * 200)))
                if self.deck:
                    z['observation'] = row['observation']
                if not self.pair["online"]:
                    continue
                by_slot[job[0]] = dict(
                    **({} if self.deck else dict(tokens=row["tokens"])),
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
