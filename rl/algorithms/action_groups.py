"""Complete-action quotient used by NOSL supervision (model width stays fixed)."""
from __future__ import annotations

import numpy as np
import torch

CONTRACT = 'equivalent-plays-v1'


def sample_groups(samples, action_dim):
    """Keep grouping with the original root; observations cannot reconstruct it."""
    identity = np.arange(action_dim, dtype=np.int16)
    groups = np.stack([s.get('action_groups', identity) for s in samples])
    masks = np.stack([s['action_mask'] for s in samples]).astype(bool)
    validate_groups(groups, masks)
    return groups.astype(np.int16)


def validate_groups(groups, masks):
    if (groups.shape != masks.shape or not np.issubdtype(groups.dtype, np.integer)
            or np.any(groups < 0) or np.any(groups >= masks.shape[-1])):
        raise ValueError('invalid complete-action groups')
    if not np.array_equal(np.take_along_axis(groups, groups, axis=-1), groups):
        raise ValueError('action groups must map directly to representatives')
    if np.any(masks & ~np.take_along_axis(masks, groups, axis=-1)):
        raise ValueError('action group representative must be legal')
    identity = np.arange(masks.shape[-1])
    if np.any((~masks) & (groups != identity)):
        raise ValueError('illegal actions cannot join a legal group')


def quotient_distribution(logits, targets, masks, groups):
    """Stable log(sum softmax) with gradients to every alias, no alias penalty.

    All reductions use float32 even under BF16 autocast. Empty groups have
    finite sentinel logits so value-only rows and zero target mass stay finite.
    """
    groups = groups.long()
    logits = logits.float().masked_fill(~masks, -1e8)
    maxima = torch.full_like(logits, -1e8).scatter_reduce(
        -1, groups, logits, reduce='amax', include_self=True).detach()
    masses = torch.zeros_like(logits).scatter_add(
        -1, groups, torch.exp(logits - maxima.gather(-1, groups)) * masks)
    group_mask = masses > 0
    group_logits = (maxima + masses.clamp_min(1e-30).log()).masked_fill(~group_mask, -1e8)
    log_probs = torch.log_softmax(group_logits, dim=-1)
    group_targets = torch.zeros_like(targets).scatter_add(-1, groups, targets)
    return log_probs, group_targets, group_mask
