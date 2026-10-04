"""Collection-indexed NOSL update budgets; no mutable scheduler clock."""
from dataclasses import asdict, dataclass
from math import cos, floor, pi
import json


@dataclass(frozen=True)
class CombatUpdateSchedule:
    mode: str = 'constant'
    peak_updates: int = 40
    hold_rollouts: int = 5
    decay_rollouts: int = 15
    final_updates: int | None = None

    def __post_init__(self):
        if self.mode not in ('constant', 'value-warmup', 'step'):
            raise ValueError('unknown NOSL update schedule')
        if any(type(x) is not int or x < 1 for x in
               (self.peak_updates, self.hold_rollouts, self.decay_rollouts)):
            raise ValueError('update schedule budgets must be positive integers')
        if self.mode == 'step':
            if type(self.final_updates) is not int or self.final_updates < 1:
                raise ValueError('step schedule requires positive integer final_updates')
        elif self.final_updates is not None:
            raise ValueError('final_updates is only supported by the step schedule')

    @classmethod
    def from_config(cls, config=None):
        return cls(**({'mode': config} if isinstance(config, str) else config or {}))

    def config(self):
        result = asdict(self)
        if self.final_updates is None:
            del result['final_updates']
        return result

    def plan(self, completed_rollouts, joint_updates):
        """Return (total updates, leading value-only updates) for the next round.

        The clock counts completed global collections, not local rank samples,
        optimizer steps, or process launches. Thus resuming does not re-warm up.
        """
        if type(completed_rollouts) is not int or completed_rollouts < 0:
            raise ValueError('completed rollouts must be a nonnegative integer')
        if type(joint_updates) is not int or joint_updates < 1:
            raise ValueError('joint updates must be a positive integer')
        if self.mode == 'constant':
            return joint_updates, 0
        if self.mode == 'step':
            return (joint_updates if completed_rollouts < self.hold_rollouts
                    else self.final_updates), 0
        if self.peak_updates < joint_updates:
            raise ValueError('warmup peak must cover the joint update budget')
        progress = min(1., max(0.,
            (completed_rollouts - self.hold_rollouts + 1) / self.decay_rollouts))
        extra = floor((self.peak_updates - joint_updates) * (1 + cos(pi * progress)) / 2 + .5)
        return joint_updates + extra, extra

    def validate_model(self, model):
        self.plan(0, int(model.nosl_is_az_updates))
        if self.mode != 'constant' and not (
            getattr(model, '_combat_distribution', False)
            and model.nosl_is_az and model.nosl_is_az_cpp and model.nosl_is_az_inner_only
            and hasattr(model.policy.core, 'policy_only_parameters')
        ):
            raise ValueError(f'{self.mode} requires entity-combat C++ inner-only NOSL')


def parse_update_schedule(value):
    """CLI accepts a preset name or a JSON object, saved in the same config form."""
    if value.lstrip().startswith('{'):
        return CombatUpdateSchedule.from_config(json.loads(value)).config()
    CombatUpdateSchedule.from_config(value)
    return value
