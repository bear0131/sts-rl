"""Numerical checks for finite logits over the same ordered legal actions.

This checks scoring equivalence, not observation completeness or sampler state.
Callers must validate those contracts separately and record their tolerances.
"""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ScoreTolerance:
    absolute: float
    relative: float
    probability: float
    kl: float

    def __post_init__(self):
        values = (self.absolute, self.relative, self.probability, self.kl)
        if any(not np.isfinite(v) or v < 0 for v in values):
            raise ValueError("score tolerances must be finite and nonnegative")


def compare_policy_scores(candidate, reference, tolerance):
    """Check per-logit errors plus softmax probability and KL errors.

    Inputs contain only legal actions, in identical order, at temperature 1.
    The reference is also a numerical implementation, not an exact oracle.
    Top-1 agreement is reported without rejecting near ties automatically.
    """
    x, y = (np.asarray(v, dtype=np.float64) for v in (candidate, reference))
    if x.ndim != 1 or x.shape != y.shape or not x.size:
        raise ValueError("expected two nonempty, identically shaped logit vectors")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("legal-action logits must be finite")

    def distribution(v):
        with np.errstate(over="ignore"):
            shifted = v - v.max()
        if not np.isfinite(shifted).all():
            raise ValueError("logit differences overflowed float64")
        logp = shifted - np.log(np.exp(shifted).sum())
        return logp, np.exp(logp)

    logp, p = distribution(x)
    logq, q = distribution(y)
    errors = np.abs(x - y)
    limits = tolerance.absolute + tolerance.relative * np.abs(y)
    scaled = np.divide(errors, limits, out=np.zeros_like(errors), where=limits > 0)
    scaled[(limits == 0) & (errors > 0)] = np.inf
    metrics = dict(
        max_abs_error=float(errors.max()),
        max_scaled_error=float(scaled.max()),
        max_probability_error=float(np.abs(p - q).max()),
        total_variation=float(np.abs(p - q).sum() / 2),
        kl_reference_to_candidate=max(0.0, float(np.sum(q * (logq - logp)))),
        top1_equal=bool(x.argmax() == y.argmax()),
    )
    failures = []
    if np.any(errors > limits):
        failures.append("logits")
    if metrics["max_probability_error"] > tolerance.probability:
        failures.append("probability")
    if metrics["kl_reference_to_candidate"] > tolerance.kl:
        failures.append("KL")
    if failures:
        raise AssertionError(f"policy score mismatch ({', '.join(failures)}): {metrics}")
    return metrics
