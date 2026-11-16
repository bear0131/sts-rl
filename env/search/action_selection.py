"""Legacy paired-search root choice, shared by rollout consumers."""
import numpy as np

SEARCH_FILTER_VALUE_TOLERANCE = .005


def turn_consensus_support(terminal_values, turn_values, action_order):
    """Veto only supported inferiority, keeping unresolved action proportions.

    Matched-world standard errors are noise heuristics, not simultaneous
    confidence bounds. Two standard errors need agreement at the natural turn
    boundary; three suffice from full returns alone. Acting still uses full Q.
    """
    order = np.asarray(action_order)
    n = len(order)
    if n < 1 or not np.array_equal(np.sort(order), np.arange(n)):
        raise ValueError('consensus requires an explicit legal action permutation')
    full = np.asarray(terminal_values, dtype=np.float64)
    if full.ndim != 1 or not np.isfinite(full).all():
        raise ValueError('consensus requires finite full-search returns')
    worlds = len(full) // n
    if worlds < 4:
        return np.ones(n, dtype=bool)
    short = np.asarray(turn_values, dtype=np.float64)
    if short.shape != full.shape or not np.isfinite(short).all():
        raise ValueError('consensus requires matching natural-turn returns')
    # Incomplete final worlds do not supply paired comparisons.
    full = full[:worlds*n].reshape(worlds, n)
    short = short[:worlds*n].reshape(worlds, n)
    diff = full[:, :, None] - full[:, None, :]
    early = short[:, :, None] - short[:, None, :]
    mean, se = diff.mean(0), diff.std(0, ddof=1) / np.sqrt(worlds)
    emean, ese = early.mean(0), early.std(0, ddof=1) / np.sqrt(worlds)
    veto = (((mean-2*se > SEARCH_FILTER_VALUE_TOLERANCE) & (emean-2*ese > 0))
            | (mean-3*se > SEARCH_FILTER_VALUE_TOLERANCE))
    keep = ~veto.any(0)
    if not keep.any():
        raise AssertionError('strict positive full-return differences must leave an action')
    result = np.empty(n, dtype=bool)
    result[order] = keep
    return result


def search_filtered_choice(logits, values, visits, nonce):
    """Q within .005 of best, then largest model score, seeded uniform ties.

    Inputs retain their public legal-action order. Return an index into that
    order and the actual tie probability, never the model sampling probability.
    """
    logits, values, visits = map(np.asarray, (logits, values, visits))
    if (logits.ndim != 1 or logits.shape != values.shape or values.shape != visits.shape
            or not np.isfinite(logits).all() or not np.isfinite(values).all()
            or not (visits > 0).any()):
        raise ValueError('search choice requires finite aligned scores and visited actions')
    eligible = np.flatnonzero((visits > 0) & (values >= values[visits > 0].max() - SEARCH_FILTER_VALUE_TOLERANCE))
    tied = eligible[logits[eligible] == logits[eligible].max()]
    if len(tied) == 1:
        return int(tied[0]), -0.
    rng = np.random.default_rng(int(nonce) ^ 0x524F4F54544945)
    return int(rng.choice(tied)), -float(np.log(len(tied)))
