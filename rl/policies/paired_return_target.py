"""Soft supervision from paired search returns, measured in HP or boss win probability."""

import numpy as np


def paired_return_target(
    world_values, *, boss, max_hp, rng, repeats=32, hp_half_gap=1.0, boss_half_gap=0.01
):
    """Bootstrap whole worlds, then average per-world-sample softmaxes of action means.

    Values use the simulator's normalized terminal objective. A partial final world
    may have missing actions (NaN). Every legal action needs at least one estimate.
    """
    values = np.asarray(world_values, dtype=np.float64)
    if (
        values.ndim != 2
        or min(values.shape) == 0
        or np.isinf(values).any()
        or not np.isfinite(values[:-1]).all()
        or not np.isfinite(values[-1]).any()
        or not np.isfinite(values).any(axis=0).all()
    ):
        raise ValueError("need paired worlds covering every action")
    if (
        not np.all(np.isfinite([max_hp, hp_half_gap, boss_half_gap]))
        or min(max_hp, hp_half_gap, boss_half_gap) <= 0
        or type(repeats) is not int
        or repeats < 1
    ):
        raise ValueError("invalid return-gap scale or bootstrap count")
    observed = np.isfinite(values)
    zero_filled = np.where(observed, values, 0.0)
    worlds = len(values)
    if worlds == 1:
        means = values
    else:
        weights = rng.multinomial(worlds, np.full(worlds, 1 / worlds), size=repeats)
        # Only a draw consisting entirely of the partial last world can miss an action.
        if not observed[-1].all():
            missing = weights[:, -1] == worlds
            while missing.any():
                weights[missing] = rng.multinomial(
                    worlds, np.full(worlds, 1 / worlds), size=int(missing.sum())
                )
                missing = weights[:, -1] == worlds
        means = (weights @ zero_filled) / (weights @ observed.astype(np.float64))
    gap = boss_half_gap if boss else hp_half_gap / max_hp
    probabilities = np.exp2((means - means.max(axis=1, keepdims=True)) / gap)
    return (probabilities / probabilities.sum(axis=1, keepdims=True)).mean(axis=0)
