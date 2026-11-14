"""Legacy paired-search root choice, shared by rollout consumers."""
import numpy as np
try:
    from slaythespire import filtered_action_ties as _native_ties
except ImportError:
    _native_ties = None


def search_filtered_choice(logits, values, visits, nonce):
    """Q within .005 of best, then largest model score, seeded uniform ties.

    Inputs retain their public legal-action order. Return an index into that
    order and the actual tie probability, never the model sampling probability.
    """
    logits, values, visits = map(np.asarray, (logits, values, visits))
    if _native_ties is not None and values.dtype == np.float64:
        tied = _native_ties(logits, values, visits)
        if len(tied) == 1:
            return tied[0], -0.
        rng = np.random.default_rng(int(nonce) ^ 0x524F4F54544945)
        return int(rng.choice(tied)), -float(np.log(len(tied)))
    if (logits.ndim != 1 or logits.shape != values.shape or values.shape != visits.shape
            or not np.isfinite(logits).all() or not np.isfinite(values).all()
            or not (visits > 0).any()):
        raise ValueError('search choice requires finite aligned scores and visited actions')
    eligible = np.flatnonzero((visits > 0) & (values >= values[visits > 0].max() - .005))
    tied = eligible[logits[eligible] == logits[eligible].max()]
    if len(tied) == 1:
        return int(tied[0]), -0.
    rng = np.random.default_rng(int(nonce) ^ 0x524F4F54544945)
    return int(rng.choice(tied)), -float(np.log(len(tied)))
