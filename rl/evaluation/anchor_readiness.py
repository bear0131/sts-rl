"""Frozen anchor value against natural future return, using the shared runner."""
import numpy as np
import torch
from stable_baselines3.common.utils import obs_as_tensor

from env.core import action_space as A
from env.core.outer_return import strategic_reward
from rl.evaluation.runner import evaluate
from rl.evaluation.strategies import PolicyStrategy


def anchor_readiness(model, scenario, config):
    if getattr(model.policy.core, 'model_contract', None) != 'deck-latent-combat-v1':
        raise ValueError('anchor readiness requires the deck latent model contract')
    model.policy.set_training_mode(False)
    result = dict(natural=0., censored=0., anchors=0., target_sum=0., target_sq_sum=0.,
                  error_sum=0., error_sq_sum=0., relic_offered=0., relic_taken=0.,
                  floor_sum=0., wins=0., timeout=0.)

    class Trace:
        def __init__(self, env, *_):
            self.env, self.total, self.anchors = env, 0., []
        def before_step(self, action, native_action):
            base = self.env.unwrapped
            self.act, self.floor = base.thick.gc.act, base.thick.gc.floor_num
            if base._at_combat_exit_anchor():
                obs = {k: v[None] for k, v in base._obs().items()}
                with torch.inference_mode():
                    value = float(model.policy.predict_values(obs_as_tensor(obs, model.device)).item())
                self.anchors.append((value, self.total))
            if base._phase() == 'combat_reward':
                mask = base.action_masks()
                start = A.seg_base('REWARD_RELIC')
                if mask[start:start+A.REWARD_RELIC].any():
                    result['relic_offered'] += 1
                    result['relic_taken'] += start <= action < start+A.REWARD_RELIC
        def after_step(self, action, reward, terminated, truncated, info, native_action):
            # The generic evaluation environment exposes its own raw reward.
            # Apply the SAME canonical strategic reward as training, once.
            self.total += strategic_reward(self.act, self.floor, self.env.unwrapped.thick.run_stats(),
                done=terminated, censored=bool(truncated or info.get('dead_screen')))
        def finish(self, outcome):
            if outcome in ('win', 'death'):
                result['natural'] += 1
                result['wins'] += outcome == 'win'
                result['floor_sum'] += self.env.unwrapped.thick.gc.floor_num
                for value, past_return in self.anchors:
                    target = self.total - past_return
                    error = target - value
                    if not np.isfinite([target, error]).all():
                        raise FloatingPointError('nonfinite actual anchor validation')
                    result['anchors'] += 1
                    result['target_sum'] += target
                    result['target_sq_sum'] += target * target
                    result['error_sum'] += error
                    result['error_sq_sum'] += error * error
            else:
                result['censored'] += 1
            return dict(anchor_samples=len(self.anchors), result=outcome)

    strategy = PolicyStrategy(model, deterministic=False)
    try:
        for _ in evaluate(strategy, scenario, config, trace_factory=Trace):
            pass
    except TimeoutError:
        result['timeout'] = 1.
    result['censored'] = max(result['censored'], config.episodes - result['natural'])
    return result


def readiness_metrics(moments):
    n = moments['anchors']
    if n < 1:
        return dict(explained_variance=None, target_std=None, mse=None)
    variance = max(0., moments['target_sq_sum'] / n - (moments['target_sum'] / n)**2)
    error_variance = max(0., moments['error_sq_sum'] / n - (moments['error_sum'] / n)**2)
    return dict(explained_variance=1 - error_variance / variance if variance > 1e-12 else None,
                target_std=variance**.5, mse=moments['error_sq_sum'] / n)
