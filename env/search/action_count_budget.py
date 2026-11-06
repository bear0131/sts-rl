"""Per-decision simulation allocation for frozen paired-world search experiments."""

from dataclasses import dataclass
import math

from env.search.conditional_subset import multi_select_slots


def effective_action_width(action_bits):
    """Count selectable cards for a subset question, otherwise legal actions."""
    if len(action_bits) == 1:
        return 1
    slots = multi_select_slots(action_bits)
    return max(2, len(slots)) if slots else len(action_bits)


def allocate_batch_budgets(widths, nominal_budget, mode, *, keys=None):
    """Redistribute one room tier's budget without changing its exact total.

    Each searched root receives between ceil(nominal/4) and 4*nominal
    simulations. Linear weights are effective action widths; tapered weights
    grow as sqrt(width) beyond eight choices. Stable root keys break rounding
    ties independently of worker completion order. Forced roots already
    bypassed by the caller do not enter this accounting.
    """
    widths = list(widths)
    if (type(nominal_budget) is not int or nominal_budget < 1
            or mode not in ("fixed", "linear", "tapered")
            or any(type(w) is not int or w < 1 for w in widths)):
        raise ValueError("invalid batch budget or action widths")
    keys = list(range(len(widths))) if keys is None else list(keys)
    if (len(keys) != len(widths) or any(type(k) is not int for k in keys)
            or len(set(keys)) != len(keys)):
        raise ValueError("need one unique integer key per root")
    if not widths or mode == "fixed":
        return [nominal_budget] * len(widths)
    weights = [float(w) if mode == "linear" else
               w / math.sqrt(max(1.0, w / 8)) for w in widths]
    lower, upper = (nominal_budget + 3) // 4, nominal_budget * 4
    total = nominal_budget * len(widths)
    lo, hi = 0.0, upper / min(weights)
    for _ in range(80):
        scale = (lo + hi) / 2
        if sum(max(lower, min(upper, scale * w)) for w in weights) < total:
            lo = scale
        else:
            hi = scale
    quotas = [max(lower, min(upper, lo * w)) for w in weights]
    result = [math.floor(q) for q in quotas]
    remaining = total - sum(result)
    order = sorted((i for i, b in enumerate(result) if b < upper),
                   key=lambda i: (-(quotas[i] - result[i]), keys[i]))
    if not 0 <= remaining <= len(order):
        raise ArithmeticError("could not conserve the simulation budget")
    for i in order[:remaining]:
        result[i] += 1
    assert sum(result) == total and all(lower <= b <= upper for b in result)
    return result


@dataclass(frozen=True)
class ActionCountBudget:
    mode: str
    regular_scale: float
    boss_scale: float

    def __post_init__(self):
        if self.mode not in ("fixed", "linear", "tapered") or not all(
            math.isfinite(x) and x > 0 for x in (self.regular_scale, self.boss_scale)
        ):
            raise ValueError("invalid action-count budget")

    def for_root(self, action_bits, boss=False):
        """Count native multi-select hand slots, rather than the combinations of them."""
        if len(action_bits) == 1:
            return 0
        return self.simulations(effective_action_width(action_bits), boss)

    def simulations(self, actions, boss=False):
        if type(actions) is not int or actions < 1:
            raise ValueError("need at least one legal action")
        if actions == 1:
            return 0
        if self.mode == "fixed":
            return 1000 if boss else 250
        scale = self.boss_scale if boss else self.regular_scale
        if self.mode == "linear":
            return actions * max(1, round(scale))
        # Cover every action. Beyond eight legal actions grow the discretionary
        # budget like sqrt(actions), while keeping a minimum number of worlds.
        floor = (32 if boss else 8) * actions
        return max(floor, round(scale * actions / math.sqrt(max(1.0, actions / 8))))


def calibrate_actions(histogram, mode, boss):
    """Match the fixed budget on recorded multi-action roots in one room tier."""
    counts = [(int(a), int(n)) for a, n in histogram.items() if a > 1 and n > 0]
    if not counts or mode not in ("linear", "tapered"):
        raise ValueError("need real multi-action decisions and an adaptive mode")
    target = (1000 if boss else 250) * sum(n for _, n in counts)
    if mode == "linear":
        return max(1, round(target / sum(a * n for a, n in counts)))

    def total(scale):
        budget = ActionCountBudget(mode, scale, scale)
        return sum(n * budget.simulations(a, boss) for a, n in counts)

    low, high = 1e-9, float(1000 if boss else 250)
    if total(low) > target:
        raise ValueError("minimum per-action coverage exceeds calibration budget")
    for _ in range(48):
        mid = (low + high) / 2
        if total(mid) > target:
            high = mid
        else:
            low = mid
    return min((low, high), key=lambda s: abs(total(s) - target))
