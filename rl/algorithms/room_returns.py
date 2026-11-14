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


def combat_anchor_returns(rewards, values, episode_starts, phases, anchors,
                          whole_run_returns, *, last_values, last_dones,
                          last_phases, last_anchors):
    """Cut only combat rows at the frozen pre-reward V, not upstream upper rows.

    Anchor rows themselves retain lambda=1 returns from the unchanged game.
    A collection cut inside an unfinished fight bootstraps its live old V;
    death uses the terminal reward. Neither case invents a successful exit.
    """
    rewards, values, targets = (np.asarray(x, dtype=np.float64) for x in
                               (rewards, values, whole_run_returns))
    shape = rewards.shape
    arrays = (values, episode_starts, phases, anchors, targets)
    if len(shape) != 2 or any(np.asarray(x).shape != shape for x in arrays):
        raise ValueError('invalid combat-anchor return arrays')
    if any(np.asarray(x).shape != (shape[1],) for x in
           (last_values, last_dones, last_phases, last_anchors)):
        raise ValueError('invalid combat-anchor return tail')
    combat_id = vocab.PHASE_VOCAB_INDEX['combat']
    anchors = np.asarray(anchors, dtype=bool)
    if np.any(anchors & (phases != combat_id)):
        raise ValueError('combat anchor must retain its combat observation phase')
    result = targets.copy()
    tail = np.asarray(last_values, dtype=np.float64)
    for t in range(shape[0] - 1, -1, -1):
        if t == shape[0] - 1:
            next_v, done, next_phase, next_anchor = last_values, last_dones, last_phases, last_anchors
        else:
            next_v, done = values[t + 1], episode_starts[t + 1]
            next_phase, next_anchor = phases[t + 1], anchors[t + 1]
        live = ~np.asarray(done, dtype=bool)
        active = (phases[t] == combat_id) & ~anchors[t]
        if np.any(active & live & (np.asarray(next_phase) != combat_id)):
            raise ValueError('combat exited without the required pre-reward anchor')
        continuation = np.where(next_anchor, next_v, tail)
        combat_target = rewards[t] + np.where(live, continuation, 0.)
        result[t] = np.where(active, combat_target, targets[t])
        tail = result[t]
    if not np.isfinite(result).all():
        raise ValueError('nonfinite combat-anchor returns')
    return result - values, result


def apply_room_returns(model, buffer, *, last_values=None):
    if last_values is None:
        from stable_baselines3.common.utils import obs_as_tensor
        with torch.no_grad(), full_precision(model.device):
            last_values = model.policy.predict_values(obs_as_tensor(model._last_obs, model.device))
    if isinstance(last_values, torch.Tensor):
        last_values = last_values.detach().flatten().cpu().numpy()
    else:
        last_values = np.asarray(last_values).reshape(-1)
    if last_values.shape != (buffer.rewards.shape[1],) or not np.isfinite(last_values).all():
        raise ValueError('invalid frozen room-return tail values')
    def ids(obs, key, scale):
        return np.rint(obs[key][..., 0] * scale).astype(np.int64)
    advantages, returns, lambdas = room_gae(
        buffer.rewards, buffer.values, buffer.episode_starts,
        ids(buffer.observations, 'act', encode.ACT_SCALE),
        ids(buffer.observations, 'floor', encode.FLOOR_SCALE),
        last_values=last_values, last_dones=model._last_episode_starts,
        last_acts=ids(model._last_obs, 'act', encode.ACT_SCALE),
        last_floors=ids(model._last_obs, 'floor', encode.FLOOR_SCALE),
        room_lambda=model.policy.room_lambda)
    phase = buffer.observations['phase'][..., 0]
    anchored = getattr(model.policy, 'combat_value_target', 'whole-run') == 'pre-reward-anchor'
    anchors = (buffer.observations['combat_exit_anchor'].any(axis=-1)
               if 'combat_exit_anchor' in buffer.observations else None)
    if anchored:
        advantages, returns = combat_anchor_returns(
            buffer.rewards, buffer.values, buffer.episode_starts, phase, anchors, returns,
            last_values=last_values, last_dones=model._last_episode_starts,
            last_phases=model._last_obs['phase'][:, 0],
            last_anchors=model._last_obs['combat_exit_anchor'].any(axis=-1))
    if anchors is not None:
        model.logger.record('pure_rl/combat_anchor_samples', int(anchors.sum()))
        model.logger.record('pure_rl/combat_anchor_wins', int(buffer.observations['combat_exit_anchor'][..., 0].sum()))
        model.logger.record('pure_rl/combat_anchor_escapes', int(buffer.observations['combat_exit_anchor'][..., 1].sum()))
    buffer.advantages[...] = advantages
    buffer.returns[...] = returns
    from rl.algorithms.ppo_metrics import value_moments
    for key, value in value_moments(buffer.values, buffer.returns, phase, anchors=anchors).items():
        model.logger.record(key, value)
    combat = phase == vocab.PHASE_VOCAB_INDEX['combat']
    if anchors is not None:
        combat = combat & ~anchors
    model.logger.record('pure_rl/combat_samples', int(combat.sum()))
    model.logger.record('pure_rl/upper_samples', int((phase != vocab.PHASE_VOCAB_INDEX['combat']).sum()))
    model.logger.record('pure_rl/room_lambda', model.policy.room_lambda)
    model.logger.record('pure_rl/in_room_edges', int((lambdas == 1).sum()))
    model.logger.record('pure_rl/boundary_edges', int(((lambdas > 0) & (lambdas < 1)).sum()))
    model.logger.record('pure_rl/search_simulations',
        getattr(model, 'deck_search_last_stats', {}).get('chains', 0)
        if getattr(model, 'deck_search_config', None) is not None else 0)
    model.logger.record('pure_rl/return_version', 2 if anchored else 1)
