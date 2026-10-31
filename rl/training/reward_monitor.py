"""Reward uptake and existing policy probabilities, without changing decisions.

A reward is resolved when claimed or left on a reward-page PROCEED. Counting
these outcomes, rather than appearances on every frame, needs no pending state
and survives collection/checkpoint boundaries through the environment itself.
"""
import numpy as np

from env.core import action_space as A, vocab as V

REWARD_PHASE = V.PHASE_VOCAB_INDEX['combat_reward']
KINDS = {
    'gold': ('reward_gold', A.seg_base('REWARD_GOLD'), A.REWARD_GOLD),
    'potion': ('reward_potion_ids', A.seg_base('REWARD_POTION'), A.REWARD_POTION),
}
FIELDS = ('resolved', 'taken', 'unavailable_left', 'policy_decisions', 'policy_probability_sum')


def counts(model):
    saved = getattr(model, '_upper_reward_counts', {})
    return {f'reward_{kind}_{field}': float(saved.get(f'reward_{kind}_{field}', 0.))
            for kind in KINDS for field in FIELDS}


def begin_collection(model):
    model._upper_reward_counts = {key: 0. for key in counts(model)}


def policy_probabilities(distribution, observations):
    """Read the already-sampled masked categorical; no extra forward or RNG."""
    indices = np.flatnonzero(np.asarray(observations['phase']).reshape(-1) == REWARD_PHASE)
    if not len(indices):
        return indices, np.empty((0, len(KINDS)), dtype=np.float32)
    import torch
    probabilities = distribution.distribution.probs
    selected = probabilities.index_select(0, torch.as_tensor(indices, device=probabilities.device))
    masses = torch.stack([selected[:, start:start + width].sum(-1)
                          for _, start, width in KINDS.values()], dim=-1)
    return indices, masses.detach().float().cpu().numpy()


def record_step(model, observations, actions, masks, policy, infos):
    indices, masses = policy
    if not len(indices):
        return
    current = counts(model)
    for row, index in enumerate(indices):
        info = infos[index]
        if (info.get('simulator_transition') is False or info.get('TimeLimit.truncated')
                or info.get('dead_screen')):
            continue
        action = int(actions[index])
        for column, (kind, (key, start, width)) in enumerate(KINDS.items()):
            prefix = f'reward_{kind}_'
            legal = np.asarray(masks[index, start:start + width], dtype=bool)
            if legal.any():
                current[prefix + 'policy_decisions'] += 1
                current[prefix + 'policy_probability_sum'] += float(masses[row, column])
            if start <= action < start + width and legal[action - start]:
                current[prefix + 'taken'] += 1
                current[prefix + 'resolved'] += 1
            elif action == A.seg_base('PROCEED'):
                remaining = np.asarray(observations[key][index]) > 0
                current[prefix + 'resolved'] += int(remaining.sum())
                current[prefix + 'unavailable_left'] += int((remaining & ~legal).sum())
    model._upper_reward_counts = current


def metrics(reduced_counts, world_size=1):
    """Reduce sums/counts together; never average rank-local percentages."""
    result = {f'reward_{kind}_{field}': reduced_counts[f'reward_{kind}_{field}'] * world_size
              for kind in KINDS for field in FIELDS}
    for kind in KINDS:
        prefix = f'reward_{kind}_'
        if result[prefix + 'resolved']:
            result[prefix + 'pickup_rate'] = result[prefix + 'taken'] / result[prefix + 'resolved']
        if result[prefix + 'policy_decisions']:
            result[prefix + 'policy_probability'] = (
                result[prefix + 'policy_probability_sum'] / result[prefix + 'policy_decisions'])
    return result
