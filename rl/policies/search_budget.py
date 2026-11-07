"""Independent-world supervision of the value and cost of root search.

Targets are signed out-of-sample action improvements, not the optimistic maximum
of the samples used to select an action. The action model receives no regression
gradient from the budget head. All constants are part of the saved mode contract.
"""
import numpy as np
import torch
from torch import nn

from rl.policies.token_regret import (
    OnlineTokenRegretPolicy, OnlineRegretSamples, pack_rows, improved_log_target,
)

BUDGET_CONTRACT = 'sts1-token64x2-independent-budget-gain-v1'
BUDGETS = (0, 32, 128, 512, 2048)
AUDIT_PROBABILITY = 1 / 64
AUDIT_WORLDS = 16
COST_PRICE = .02  # HP units (boss percentage points) per 1000 transitions.
EXPLORATION = .2
WARMUP_ROUNDS = 3
WARMUP_PROBABILITIES = np.array((.05, .15, .4, .3, .1))


def budget_manifest():
    return dict(contract=BUDGET_CONTRACT, budgets=list(BUDGETS),
                audit_probability=AUDIT_PROBABILITY, audit_worlds=AUDIT_WORLDS,
                cost_price_per_1000_transitions=COST_PRICE,
                exploration=EXPLORATION, warmup_rounds=WARMUP_ROUNDS,
                warmup_probabilities=WARMUP_PROBABILITIES.tolist())


def choose_budget(prediction, nonce, *, training, round_index):
    prediction = np.asarray(prediction, np.float64)
    if prediction.shape != (8,) or not np.isfinite(prediction).all():
        raise ValueError('budget predictor must return four finite gains and costs')
    rng = np.random.default_rng(int(nonce) ^ 0x425544474554)
    utility = np.r_[0., prediction[:4] * 10 - COST_PRICE *
                    np.expm1(np.clip(prediction[4:], 0., 15.))]
    if training and round_index < WARMUP_ROUNDS:
        return int(rng.choice(len(BUDGETS), p=WARMUP_PROBABILITIES))
    if training and rng.random() < EXPLORATION:
        return int(rng.integers(len(BUDGETS)))
    # The cheaper choice wins exact utility ties, including the zero anchor.
    return int(utility.argmax())


class BudgetSamples(OnlineRegretSamples):
    def __init__(self, rows):
        self.rows = rows
        self.log_targets = []
        for row in rows:
            if row['budget_searched']:
                target = improved_log_target(row)
            else:
                raw = np.asarray(row['prior_logits'], np.float64)
                target = raw - np.logaddexp.reduce(raw)
            self.log_targets.append(target.astype(np.float32))
        self.teacher = np.array([t.argmax() for t in self.log_targets])
        self.target_entropy = sum(float(-(np.exp(t[np.isfinite(t)]) *
                                         t[np.isfinite(t)]).sum()) for t in self.log_targets)
        self.uniform_ce = sum(np.log(len(t)) for t in self.log_targets)

    def batch(self, indices, device):
        inputs, target, teacher = super().batch(indices, device)
        rows = [self.rows[i] for i in indices]
        extra = np.array([[float(r['budget_searched']), float(r['budget_audit']),
                           float(r['budget_boss']), *r['budget_targets']] for r in rows], np.float32)
        return (*inputs[:-1], torch.as_tensor(extra, device=device), inputs[-1]), target, teacher


class AdaptiveTokenRegretPolicy(OnlineTokenRegretPolicy):
    def __init__(self, key_count):
        super().__init__(key_count)
        self.budget_head = nn.Sequential(nn.Linear(100, 64), nn.ReLU(), nn.Linear(64, 8))
        nn.init.zeros_(self.budget_head[-1].weight)
        nn.init.zeros_(self.budget_head[-1].bias)
        with torch.no_grad():
            self.budget_head[-1].bias[4:].copy_(torch.tensor(
                np.log1p(np.array(BUDGETS[1:]) * 32 / 1000), dtype=torch.float32))
        self.register_buffer('budget_round', torch.zeros((), dtype=torch.int64))

    def collection_pair(self, *, collect_samples):
        pair = super().collection_pair(collect_samples=collect_samples)
        pair['budget_config'] = budget_manifest()
        pair['budget_round'] = int(self.budget_round)
        return pair

    def predict_budget(self, context, state, scores, mask, boss):
        lp = scores.log_softmax(-1).masked_fill(~mask, 0.)
        p = lp.exp().masked_fill(~mask, 0.)
        count = mask.sum(-1).float()
        entropy = -(p * lp).sum(-1) / count.clamp_min(2).log()
        top = scores.topk(min(2, scores.shape[-1]), -1).values
        gap = top[:, 0] - top[:, -1]
        gap = torch.where(count > 1, gap, torch.zeros_like(gap)).clamp(0, 20) / 20
        summaries = torch.stack((entropy, gap, count.log1p() / 10, boss.float()), -1)
        return self.budget_head(torch.cat((context, state, summaries), -1).detach())

    def score_with_budget(self, inputs, boss):
        full, guide, context = self.scores_and_guide(*inputs, return_context=True)
        return full, self.predict_budget(context, inputs[0], full, inputs[-1], boss)

    def prepare_samples(self, samples, target_mix):
        return BudgetSamples(samples)

    def update_terms(self, inputs, log_target):
        extra = inputs[-2]
        base = (*inputs[:8], inputs[-1])
        full, guide, context = self.scores_and_guide(*base, return_context=True)
        observed = torch.isfinite(log_target)
        lp = full.masked_fill(~observed, -torch.inf).log_softmax(-1).masked_fill(~observed, 0.)
        target = log_target.masked_fill(~observed, 0.)
        p = lp.exp().masked_fill(~observed, 0.)
        reverse = ((p * (lp - target)).sum(-1) * extra[:, 0]).sum()
        forward = ((log_target.exp() * (target - lp)).sum(-1) * extra[:, 0]).sum()
        legal = base[-1]
        root_lp = full.log_softmax(-1).masked_fill(~legal, 0.)
        guide_lp = guide.log_softmax(-1).masked_fill(~legal, 0.)
        teacher = root_lp.detach()
        guide_kl = (teacher.exp().masked_fill(~legal, 0.) * (teacher - guide_lp)).sum()
        entropy = -(root_lp.exp() * root_lp * legal).sum()
        prediction = self.predict_budget(context, base[0], full, legal, extra[:, 2])
        squared = (prediction - extra[:, 3:]).square()
        gain_loss = (squared[:, :4].mean(-1) * extra[:, 1]).sum() / AUDIT_PROBABILITY
        cost_loss = (squared[:, 4:].mean(-1) * extra[:, 1]).sum() / AUDIT_PROBABILITY
        loss = reverse + .5 * guide_kl + gain_loss + .1 * cost_loss
        metrics = dict(budget_gain_mse=gain_loss, budget_cost_log_mse=cost_loss,
                       budget_audit_fraction=extra[:, 1].sum())
        return loss, entropy, root_lp, forward, guide_kl, metrics
