"""Read-only main campfire choices; follow-up selections are other phases."""
import numpy as np

from env.core import action_space as A, game_actions as G, vocab as V

REST_PHASE = V.PHASE_VOCAB_INDEX['rest']
OPTIONS = {name: A.seg_base('CAMPFIRE') + index
           for name, index in G.CAMPFIRE_OPTIONS.items()}
OPTIONS['skip'] = A.seg_base('PROCEED')
FIELDS = ('available', 'selected', 'probability_sum')


def counts(model):
    saved = getattr(model, '_upper_campfire_counts', {})
    keys = ['campfire_decisions'] + [f'campfire_{name}_{field}'
                                    for name in OPTIONS for field in FIELDS]
    return {key: float(saved.get(key, 0.)) for key in keys}


def begin_collection(model):
    model._upper_campfire_counts = {key: 0. for key in counts(model)}


def policy_probabilities(distribution, observations):
    """Read the existing masked distribution, conditional on a main choice.

    Potion actions can also be legal at a campfire. Their probability mass
    must not enter the comparison with actual committed campfire choices.
    """
    indices = np.flatnonzero(np.asarray(observations['phase']).reshape(-1) == REST_PHASE)
    if not len(indices):
        return indices, np.empty((0, len(OPTIONS)), dtype=np.float32)
    import torch
    probs = distribution.distribution.probs
    selected = probs.index_select(0, torch.as_tensor(indices, device=probs.device))
    selected = selected[:, list(OPTIONS.values())].detach().float()
    selected = selected / selected.sum(-1, keepdim=True).clamp_min(torch.finfo(torch.float32).tiny)
    return indices, selected.cpu().numpy()


def record_step(model, actions, masks, policy, infos):
    indices, probabilities = policy
    if not len(indices):
        return
    current = counts(model)
    for row, index in enumerate(indices):
        info = infos[index]
        action = int(actions[index])
        if (info.get('simulator_transition') is False or info.get('TimeLimit.truncated')
                or info.get('dead_screen') or action not in OPTIONS.values()
                or not masks[index, action]):
            continue
        current['campfire_decisions'] += 1
        for column, (name, option) in enumerate(OPTIONS.items()):
            if masks[index, option]:
                prefix = f'campfire_{name}_'
                current[prefix + 'available'] += 1
                current[prefix + 'selected'] += int(action == option)
                current[prefix + 'probability_sum'] += float(probabilities[row, column])
    model._upper_campfire_counts = current


def metrics(reduced_counts, world_size=1):
    """Scalar reduction returns rank means; restore sums before dividing."""
    result = {key: reduced_counts[key] * world_size for key in counts(None)}
    for name in OPTIONS:
        prefix = f'campfire_{name}_'
        if result['campfire_decisions']:
            result[prefix + 'share'] = result[prefix + 'selected'] / result['campfire_decisions']
        if result[prefix + 'available']:
            result[prefix + 'choice_rate'] = result[prefix + 'selected'] / result[prefix + 'available']
            result[prefix + 'policy_probability'] = (
                result[prefix + 'probability_sum'] / result[prefix + 'available'])
    return result
