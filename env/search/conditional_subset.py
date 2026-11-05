"""Conditional yes/no decisions for a complete native multi-card selection."""

import numpy as np


def multi_select_slots(action_bits):
    bits = np.asarray(action_bits, dtype=np.uint32)
    if bits.ndim != 1 or not len(bits):
        raise ValueError("empty or malformed legal actions")
    multi = bits >> 29 == 3
    if not multi.any():
        return ()
    if not multi.all() or len(np.unique(bits)) != len(bits):
        raise ValueError("expected one set of unique native multi-select actions")
    masks = bits & 0xFFFF
    used = int(np.bitwise_or.reduce(masks))
    if used >= 1 << 10:
        raise ValueError("multi-selection outside native hand slots")
    return tuple(i for i in range(10) if used & (1 << i))


def conditional_probabilities(action_bits, joint, order=None):
    """Return teacher mass and P(include | previous real slots), for positive-mass prefixes."""
    bits = np.asarray(action_bits, dtype=np.uint32)
    slots = multi_select_slots(bits)
    if order is not None:
        if sorted(order) != list(slots):
            raise ValueError("slot order must cover each selectable card once")
        slots = tuple(order)
    p = np.asarray(joint, dtype=np.float64)
    if (
        p.shape != bits.shape
        or not np.isfinite(p).all()
        or (p < 0).any()
        or not np.isclose(p.sum(), 1)
    ):
        raise ValueError("invalid joint target")
    output = {}
    prefix_mask = 0
    for slot in slots:
        seen = bits & prefix_mask
        take = (bits & (1 << slot)) != 0
        for prefix in np.unique(seen[p > 0]):
            active = seen == prefix
            weight = p[active].sum()
            output[(slot, int(prefix))] = (
                float(weight),
                float(p[active & take].sum() / weight),
            )
        prefix_mask |= 1 << slot
    return output


def search_subset(action_bits, budget, evaluate, rng, prior_logits, band=0.005):
    """Compare skip/take with paired completions, then commit one legal final subset.

    evaluate(indices, n, nonce) returns a [world, sampled action] matrix of normalized
    returns; the last world can be partial. No simulation changes the real root.
    """
    bits = np.asarray(action_bits, dtype=np.uint32)
    slots = multi_select_slots(bits)
    if not slots or type(budget) is not int or budget < 2 * len(slots):
        raise ValueError(
            "need a multi-card root and enough budget to compare both branches"
        )
    logits = np.asarray(prior_logits, np.float64)
    if logits.shape != bits.shape or not np.isfinite(logits).all():
        raise ValueError("incompatible subset prior")
    prior = np.exp(logits - logits.max())
    active = np.arange(len(bits))
    visits = np.zeros(len(bits), np.int64)
    remaining = budget
    for j, slot in enumerate(slots):
        if len(active) == 1:
            break
        take = (bits[active] & (1 << slot)) != 0
        branches = [active[~take], active[take]]
        if not all(len(group) for group in branches):
            continue
        # Pair common completion quantiles. Stratified ranks are distinct within a branch.
        allotted = remaining // (len(slots) - j)
        m = min(8, len(branches[0]), len(branches[1]), allotted // 2)
        jitter = rng.random(m)
        sampled_groups = []
        for group in branches:
            boundaries = np.arange(m + 1) * len(group) // m
            ranks = boundaries[:-1] + (jitter * np.diff(boundaries)).astype(int)
            sampled_groups.append(group[ranks])
        sampled = np.stack(sampled_groups)
        # Both branches share every sampled world. Alternate any odd remaining visit.
        order = rng.permutation(2 * m)
        indices = sampled.ravel()[order]
        values = np.asarray(
            evaluate(indices, allotted, int(rng.integers(2**63))), np.float64
        )
        if (
            values.ndim != 2
            or values.shape[1] != len(indices)
            or np.isfinite(values).sum() != allotted
        ):
            raise ValueError("incomplete paired branch evaluation")
        counts = np.isfinite(values).sum(0)
        visits[indices] += counts
        sums = np.nansum(values, axis=0)
        is_take = order >= m
        q = np.array(
            [
                sums[is_take == x].sum() / counts[is_take == x].sum()
                for x in (False, True)
            ]
        )
        if abs(q[0] - q[1]) <= band:
            mass = np.array([prior[group].sum() for group in branches])
            choose = int(rng.choice(np.flatnonzero(mass == mass.max())))
        else:
            choose = int(q.argmax())
        active = branches[choose]
        remaining -= allotted
    if len(active) != 1:
        raise ValueError("ambiguous final subset")
    return int(active[0]), visits
