"""Per-decision simulation allocation for frozen paired-world search experiments."""

from dataclasses import dataclass
import math

from env.search.conditional_subset import multi_select_slots


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
        slots = multi_select_slots(action_bits)
        width = max(2, len(slots)) if slots else len(action_bits)
        return self.simulations(width, boss)

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
