"""Frozen anchor value against natural future return, using the shared runner."""
import numpy as np
import torch
from stable_baselines3.common.utils import obs_as_tensor

from env.core import action_space as A
from env.core import encode
from env.core.outer_return import strategic_reward
from rl.evaluation.runner import evaluate
from rl.evaluation.strategies import PolicyStrategy


@torch.inference_mode()
def anchor_hp_response(model, observation, *, hp, max_hp):
    """Hold a real pre-reward anchor fixed and vary only current-HP inputs.

    This measures the frozen network, not simulated counterfactual survival.
    Raw HP comes from BattleContext: encoded HP/maxHP saturate at HP_SCALE.
    """
    if not observation['combat_exit_anchor'].any() or not 1 <= hp <= max_hp:
        raise ValueError('HP response requires a live pre-reward anchor')
    levels = np.unique(np.r_[np.rint(np.linspace(1, max_hp, 21)).astype(int),
        hp, max(1, hp-1), min(max_hp, hp+1), max(1, hp-5), min(max_hp, hp+5)])
    batch = {k: np.repeat(v[None], len(levels), axis=0) for k, v in observation.items()}
    batch['player_scalars'][:, 0] = np.minimum(levels, encode.HP_SCALE) / encode.HP_SCALE
    batch['player_scalars'][:, 5] = levels / max_hp
    batch['run_hp'][:, 0] = batch['player_scalars'][:, 0]
    batch['run_hp'][:, 2] = levels / max_hp
    values = model.policy.predict_values(obs_as_tensor(batch, model.device)).flatten().cpu().numpy()
    if not np.isfinite(values).all():
        raise FloatingPointError('nonfinite anchor HP response')
    return dict(hp=int(hp), max_hp=int(max_hp), hp_levels=levels.tolist(), values=values.tolist(),
                actual_value=float(values[np.flatnonzero(levels == hp)[0]]))


def anchor_readiness(model, scenario, config, *, hp_response=False):
    if getattr(model.policy.core, 'model_contract', None) != 'deck-latent-combat-v1':
        raise ValueError('anchor readiness requires the deck latent model contract')
    model.policy.set_training_mode(False)
    result = dict(natural=0., censored=0., anchors=0., target_sum=0., target_sq_sum=0.,
                  error_sum=0., error_sq_sum=0., relic_offered=0., relic_taken=0.,
                  floor_sum=0., wins=0., timeout=0.)
    if hp_response:
        result['hp_response'] = dict(contract='fixed-anchor-current-hp-v1', rows=[],
            scope='natural-game anchors; network input response, not counterfactual win rates')

    class Trace:
        def __init__(self, env, episode, seed):
            self.env, self.total, self.anchors = env, 0., []
            self.episode, self.seed = episode, seed
        def before_step(self, action, native_action):
            base = self.env.unwrapped
            self.act, self.floor = base.thick.gc.act, base.thick.gc.floor_num
            if base._at_combat_exit_anchor():
                obs = base._obs()
                response = None
                if hp_response:
                    player = base.thick.bc.to_dict()['player']
                    response = anchor_hp_response(model, obs, hp=int(player['hp']), max_hp=int(player['max_hp']))
                    response.update(episode=self.episode, seed=self.seed, act=int(self.act), floor=int(self.floor))
                    value = response['actual_value']
                else:
                    with torch.inference_mode():
                        value = float(model.policy.predict_values(obs_as_tensor(
                            {k: v[None] for k, v in obs.items()}, model.device)).item())
                self.anchors.append((value, self.total, response))
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
                for value, past_return, response in self.anchors:
                    target = self.total - past_return
                    error = target - value
                    if not np.isfinite([target, error]).all():
                        raise FloatingPointError('nonfinite actual anchor validation')
                    result['anchors'] += 1
                    result['target_sum'] += target
                    result['target_sq_sum'] += target * target
                    result['error_sum'] += error
                    result['error_sq_sum'] += error * error
                    if response is not None:
                        result['hp_response']['rows'].append(dict(response, target=target, outcome=outcome))
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
