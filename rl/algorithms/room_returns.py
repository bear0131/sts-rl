"""Whole-run GAE: propagate through all actions, decay only between rooms."""
import numpy as np
import torch

from env.core import encode, vocab
from rl.policies.precision import full_precision


def room_gae(rewards, values, episode_starts, acts, floors, *, last_values,
             last_dones, last_acts, last_floors, room_lambda):
    """Gamma=1. Time-limit bootstrap is already included by the PPO collector.

    Act/floor jointly identify a native room visit. A new Act is one boundary,
    never a terminal or a second multiplicative decay. No future reward is used
    to choose lambda. Adjacent behavior values cancel exactly at lambda=1.
    """
    rewards, values = (np.asarray(x, dtype=np.float64) for x in (rewards, values))
    shape = rewards.shape
    if (len(shape) != 2 or values.shape != shape
            or any(np.asarray(x).shape != shape for x in (episode_starts, acts, floors))
            or not np.isfinite(room_lambda) or not 0 <= room_lambda <= 1):
        raise ValueError('invalid room-return arrays or lambda')
    if any(np.asarray(x).shape != (shape[1],) for x in
           (last_values, last_dones, last_acts, last_floors)):
        raise ValueError('invalid room-return tail')
    advantages, lambdas = np.zeros(shape, np.float64), np.zeros(shape, np.float64)
    tail = np.zeros(shape[1], np.float64)
    for t in range(shape[0] - 1, -1, -1):
        if t == shape[0] - 1:
            next_v, done, next_act, next_floor = last_values, last_dones, last_acts, last_floors
        else:
            next_v, done = values[t + 1], episode_starts[t + 1]
            next_act, next_floor = acts[t + 1], floors[t + 1]
        live = ~np.asarray(done, dtype=bool)
        same_room = (acts[t] == next_act) & (floors[t] == next_floor)
        lam = np.where(live, np.where(same_room, 1., room_lambda), 0.)
        # where avoids contaminating terminals with an irrelevant next V.
        delta = rewards[t] + np.where(live, next_v, 0.) - values[t]
        tail = delta + lam * tail
        advantages[t], lambdas[t] = tail, lam
    returns = advantages + values
    if not (np.isfinite(returns).all() and np.isfinite(advantages).all()):
        raise ValueError('nonfinite whole-run returns')
    return advantages, returns, lambdas


def apply_room_returns(model, buffer):
    from stable_baselines3.common.utils import obs_as_tensor
    with torch.no_grad(), full_precision(model.device):
        last_values = model.policy.predict_values(obs_as_tensor(model._last_obs, model.device))
    def ids(obs, key, scale):
        return np.rint(obs[key][..., 0] * scale).astype(np.int64)
    advantages, returns, lambdas = room_gae(
        buffer.rewards, buffer.values, buffer.episode_starts,
        ids(buffer.observations, 'act', encode.ACT_SCALE),
        ids(buffer.observations, 'floor', encode.FLOOR_SCALE),
        last_values=last_values.flatten().cpu().numpy(), last_dones=model._last_episode_starts,
        last_acts=ids(model._last_obs, 'act', encode.ACT_SCALE),
        last_floors=ids(model._last_obs, 'floor', encode.FLOOR_SCALE),
        room_lambda=model.policy.room_lambda)
    buffer.advantages[...] = advantages
    buffer.returns[...] = returns
    phase = buffer.observations['phase'][..., 0]
    model.logger.record('pure_rl/combat_samples', int((phase == vocab.PHASE_VOCAB_INDEX['combat']).sum()))
    model.logger.record('pure_rl/upper_samples', int((phase != vocab.PHASE_VOCAB_INDEX['combat']).sum()))
    model.logger.record('pure_rl/room_lambda', model.policy.room_lambda)
    model.logger.record('pure_rl/in_room_edges', int((lambdas == 1).sum()))
    model.logger.record('pure_rl/boundary_edges', int(((lambdas > 0) & (lambdas < 1)).sum()))
    model.logger.record('pure_rl/search_simulations', 0)
    model.logger.record('pure_rl/return_version', 1)
