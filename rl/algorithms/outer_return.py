"""Executed combat segments under a fixed upper actor/critic (gamma = 1)."""
from functools import lru_cache
import hashlib
from pathlib import Path

import numpy as np
import torch

from env.core import encode, vocab as V
from env.core.outer_return import OBJECTIVE, CONTEXT_SCHEMA, CombatExit, strategic_reward, SCALAR_OBJECTIVES, FINITE_OBJECTIVE, OUTER_OBJECTIVES
from env.core.finite_curves import FACTORED_OBJECTIVE, CURVE_OBJECTIVES, winning_exit_target
from rl.search.native import observation_as_tensor
from rl.algorithms.combat_distribution import MEAN_CONTRACT

COMBAT = V.PHASE_VOCAB_INDEX['combat']


@lru_cache(maxsize=1)
def _native_sha():
    import slaythespire as sts
    with Path(sts.__file__).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def contract(model):
    teacher = model.policy.core.teacher
    objective = model.policy.core.training_objective
    identity = dict(objective=objective, context=CONTEXT_SCHEMA, gamma=1.,
                lower_initialization=model.policy.core.lower_initialization,
                teacher=teacher.get_extra_state(), native_sha256=_native_sha(),
                observation_version=encode.OBS_VERSION, boundary='settled-run-exit-v1',
                search={name: getattr(model, 'nosl_is_az_' + name) for name in
                        ('simulations', 'depth', 'fixed_worldline', 'search_mode',
                         'paper_full_gumbel', 'gumbel_candidates', 'gumbel_scale',
                         'gumbel_value_scale', 'gumbel_rescale_q', 'gumbel_use_mixed_value',
                         'gumbel_max_visit_init', 'merge_equivalent_actions')})
    if objective in CURVE_OBJECTIVES:
        identity['finite_curves'] = model.policy.core.finite_curve_bank.identity
        if objective == FACTORED_OBJECTIVE:
            identity.update(value_head='win-probability-and-conditional-curve-mean-v1',
                            loss='win-bce-plus-win-masked-curve-mse-v1',
                            non_victory='loss-and-player-escape-zero-v1')
    elif objective != OBJECTIVE:
        config = model.policy.core.config
        identity.update(boundary='combat-end-hp-v1', boss_win_weight=config.boss_win_weight,
                        mean_loss_coef=config.mean_loss_coef, auxiliary_loss_coef=config.auxiliary_loss_coef)
        if objective == MEAN_CONTRACT:
            identity.update(value_head='win-probability-and-conditional-hp-mean-v1',
                            loss='win-bce-plus-win-masked-hp-mse-v1')
        else:
            identity['quantiles'] = config.quantiles
    return identity


def setup(model):
    if not (model.nosl_is_az and model.nosl_is_az_inner_only and model.nosl_is_az_cpp and
            model.nosl_is_az_upper_strategy == 'model' and model.nosl_is_az_combat_strategy == 'model' and
            model.nosl_is_az_fixed_worldline and not model.nosl_is_az_boss_transition_target and
            not model.nosl_is_az_end_turn_as_leaf and (not model._outer_return or not model.hierarchical_combat_hp) and
            model.nosl_is_az_leaf_batch_size == 1 and model.gamma == 1.):
        raise ValueError('outer return requires gamma=1, fixed-worldline native inner-only NOSL and frozen model upper')
    if model.get_env() is not None and getattr(model.get_env(), 'native_search_batch', None) is None:
        raise ValueError('outer return requires resident native vector environments')
    identity = contract(model)
    previous = getattr(model, 'outer_return_contract', None)
    if previous is not None and previous != identity:
        raise ValueError('outer-return checkpoint teacher/context/native identity changed')
    model.outer_return_contract = identity
    model.nosl_value_contract = model.policy.core.training_objective if model._outer_return else 'combat-end-hp-v1'
    model.policy.core.teacher.assert_integrity()
    if not all(parameter.requires_grad for name, parameter in model.policy.core.combat.named_parameters()
               if not (not model._outer_return and not model.policy.core.config.auxiliary_loss_coef
                       and name.startswith('transition.'))):
        raise ValueError('outer return trains all lower actor/encoder/critic parameters from the first update')


def begin_collection(model, count):
    model.policy.core.teacher.assert_integrity()
    rewards = getattr(model, '_outer_pending_rewards', None)
    if rewards is None:
        model._outer_pending_rewards = np.zeros(count, dtype=np.float64)
    elif np.shape(rewards) != (count,):
        raise ValueError('outer-return pending environment count changed')
    model._outer_collection = dict(upper_exits=0, combat_exits=0, terminal_exits=0,
                                   censored_segments=0, labelled_roots=0, target_sum=0.,
                                   target_min=None, target_max=None, upper_updates=0)
    validate_pending(model, count)


def validate_pending(model, count):
    rewards = np.asarray(getattr(model, '_outer_pending_rewards', None))
    pending = getattr(model, '_nosl_is_az_pending_fights', None)
    if (rewards.shape != (count,) or not np.isfinite(rewards).all() or
            not isinstance(pending, list) or len(pending) != count):
        raise ValueError('invalid outer-return pending state')
    for i, samples in enumerate(pending):
        for sample in samples:
            prefix = sample.get('outer_reward_prefix')
            if prefix is None or not np.isfinite(prefix) or prefix > rewards[i]:
                raise ValueError('invalid outer-return pending prefix')


def stamp_roots(model, pending):
    for i, samples in enumerate(pending):
        if samples:
            samples[-1].setdefault('outer_reward_prefix', float(model._outer_pending_rewards[i]))


def upper_actions(model, observations, masks, actions):
    indices = np.flatnonzero(np.asarray(observations['phase']).reshape(-1) != COMBAT)
    if len(indices):
        obs = observation_as_tensor({key: value[indices] for key, value in observations.items()}, model.device)
        decisions, _ = model.policy.core.teacher.decide(obs, masks[indices])
        actions[indices] = decisions.cpu().numpy()


def finish_step(model, before, after, dones, infos, pending):
    """Label only executed roots, including roots retained across collections."""
    completed = discarded = fights = 0
    bank = getattr(model.policy.core, 'finite_curve_bank', None)
    factored = getattr(model.policy.core, 'training_objective', None) == FACTORED_OBJECTIVE
    exits = {}
    for i, info in enumerate(infos):
        if int(before['phase'][i, 0]) != COMBAT:
            continue
        censored = bool(info.get('TimeLimit.truncated') or info.get('dead_screen'))
        stats = info.get('run_stats')
        if stats is None:
            raise ValueError('outer-return collection needs actual run statistics')
        model._outer_pending_rewards[i] += strategic_reward(
            round(float(before['act'][i, 0]) * encode.ACT_SCALE),
            round(float(before['floor'][i, 0]) * encode.FLOOR_SCALE),
            stats, done=bool(dones[i]), censored=censored)
        if not (dones[i] or int(after['phase'][i, 0]) != COMBAT or info.get('outer_combat_exit')):
            continue
        if censored:
            discarded += len(pending[i])
            pending[i] = []
            model._outer_pending_rewards[i] = 0.
            model._outer_collection['censored_segments'] += 1
            continue
        kind = 'terminal' if dones[i] else ('combat' if int(after['phase'][i, 0]) == COMBAT else 'upper')
        exits[i] = CombatExit(float(model._outer_pending_rewards[i]), kind, info.get('finite_curve_exit'))
    bootstraps = {}
    for kind in ('upper', 'combat'):
        indices = [i for i, exit in exits.items() if exit.kind == kind]
        if not indices:
            continue
        if bank is not None:
            for i in indices:
                state = infos[i].get('finite_curve_exit')
                if state is None:
                    raise ValueError('finite curve collection omitted terminal encounter/resources')
                if not factored:
                    bootstraps[i] = bank.value(state)
            continue
        obs = observation_as_tensor({key: value[indices] for key, value in after.items()}, model.device)
        with torch.no_grad():
            if kind == 'upper':
                _, values = model.policy.core.teacher(obs)
            else:
                with model.autocast_context():
                    _, values = model.policy.core.combat_actor_critic(obs)
        bootstraps.update(zip(indices, values.float().flatten().cpu().numpy(), strict=True))
    for i, exit in exits.items():
        if factored:
            from rl.algorithms.combat_distribution import TARGET_WIDTH
            win, target = winning_exit_target(exit, bank)
        else:
            target = exit.target(bootstraps.get(i))
        for sample in pending[i]:
            if factored:
                value = target - sample['outer_reward_prefix'] if win else 0.
                sample['combat_targets'] = np.zeros(TARGET_WIDTH, dtype=np.float32)
                sample['combat_targets'][0] = win
            else:
                value = target - sample['outer_reward_prefix']
            if not np.isfinite(value):
                raise ValueError('outer-return target is not finite')
            sample['value_target'] = value
            sample['outer_exit_kind'] = exit.kind
            sample['outer_episode_seed'] = infos[i]['run_stats'].get('seed')
            sample['outer_outcome'] = infos[i]['run_stats']['outcome']
            metrics = model._outer_collection
            metrics['target_sum'] += value
            metrics['target_min'] = value if metrics['target_min'] is None else min(metrics['target_min'], value)
            metrics['target_max'] = value if metrics['target_max'] is None else max(metrics['target_max'], value)
        completed += len(pending[i])
        model._nosl_is_az_samples.extend(pending[i])
        pending[i] = []
        model._outer_pending_rewards[i] = 0.
        model._outer_collection[exit.kind + '_exits'] += 1
        fights += 1
    model._outer_collection['labelled_roots'] += completed
    return completed, discarded, fights


def finish_collection(model):
    model.policy.core.teacher.assert_integrity()
    for name, value in model._outer_collection.items():
        # Every rank emits identical keys, including an empty local collection.
        model.logger.record('outer_return/' + name, 0. if value is None else value)
