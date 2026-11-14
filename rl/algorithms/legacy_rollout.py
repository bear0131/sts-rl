"""The stepfull480 ready/search/upper/receive/step loop with a PPO consumer.

A logical environment supplies exactly n_rollout_steps rows. Completion order
never changes its paired worlds or policy draws. No vector search barrier,
per-group process transport, or speculative real transition is introduced.
"""
import time
import numpy as np
import torch
from stable_baselines3.common.vec_env import VecMonitor

from env.core import vocab
from rl.algorithms.legacy_search_stream import Stream
from rl.algorithms.room_returns import apply_room_returns
from rl.training import reward_monitor, campfire_monitor


def collect(model, env, callback, rollout_buffer, n_rollout_steps, use_masking=True):
    from rl.training.native_vec_env import NativeSearchVecEnv
    from rl.algorithms.search_auxiliary import SearchRolloutBuffer, SearchTrainingConfig
    if (not isinstance(env, VecMonitor) or type(env.venv) is not NativeSearchVecEnv
            or not isinstance(rollout_buffer, SearchRolloutBuffer) or not use_masking
            or model.device.type != 'cuda' or model.amp_mode != 'off'
            or model.nosl_supervision_contract != 'shared-wholerun-ppo-room-v1'
            or model.reasoning_updates or model.reasoning_coef or model.use_sde
            or getattr(model, 'episode_rollout_config', None)):
        raise ValueError('legacy search collection requires fixed FP32 native whole-run PPO')
    config = SearchTrainingConfig(**model.deck_search_config)
    if config.action_selection != 'search-filter-policy-rank-v1' or config.backend != 'cuda-resident':
        raise ValueError('legacy collector requires explicit search-filter/model-rank CUDA behavior')
    if n_rollout_steps != rollout_buffer.buffer_size:
        raise ValueError('rollout must fill its declared sample count')
    vec = env.venv
    mask_getters = [e.get_wrapper_attr('action_masks') for e in vec.envs]
    import slaythespire as sts
    if not hasattr(sts, 'ObservationRowCopy'):
        raise RuntimeError('legacy PPO consumer requires the explicit native row-copy build')
    if any(not np.array_equal(value, vec.buf_obs[key]) for key,value in model._last_obs.items()):
        raise RuntimeError('restored policy observation differs from live vector observation')
    # The collector owns these slots; save each pre-action row before advancing
    # it. No intermediate full-observation copy or copy-back is necessary.
    model._last_obs = vec.buf_obs
    vector_start = dict(vec.step_seconds)
    model.policy.set_training_mode(False)
    rollout_buffer.reset()
    observation_copy = sts.ObservationRowCopy(rollout_buffer.observations, vec.buf_obs, 2)
    reward_monitor.begin_collection(model)
    campfire_monitor.begin_collection(model)
    rng = np.random.default_rng(config.seed)
    if getattr(model, 'deck_search_rng', None) is not None:
        rng.bit_generator.state = model.deck_search_rng
    draws = rng.random((n_rollout_steps, vec.num_envs, 2))
    seeds = rng.integers(0, 2**63-1, (n_rollout_steps, vec.num_envs), dtype=np.int64)
    positions = rollout_buffer.slot_positions
    ready = set(range(vec.num_envs))
    upper_waiting, actions, masks, estimates, upper_policy = {}, {}, {}, {}, {}
    deferred_values = []
    upper_pending = None
    action_deadline = None
    combat = vocab.PHASE_VOCAB_INDEX['combat']
    runic_dome = vocab.RELIC_VOCAB_INDEX['Runic Dome']
    stats = dict(policy_seconds=0., env_step_seconds=0., search_wait_seconds=0.,
        root_submit_seconds=0., root_prepare_seconds=0.,
        upper_batches=0, upper_rows=0, transition_batches=0, transitions=0,
        simulator_transitions=0, selection_prefix_decisions=0,
        transitions_with_upper_pending=0, transitions_with_fallback_pending=0,
        value_rows=0, value_seconds=0., forced_rows=0)
    model._independent_rollout_progress = stats
    started = time.monotonic()
    stream = None
    previous_value_observations = [e.unwrapped._emit_nosl_value_observations for e in vec.envs]
    for e in vec.envs:
        e.unwrapped._emit_nosl_value_observations = False
    model._reasoning_rollout_samples = []

    def consume_upper():
        nonlocal upper_pending
        if upper_pending is None:
            return
        ident, rows = upper_pending
        completed, packet = stream.service.take_reply(ident)
        if not completed:
            return
        reward = dict(zip(*packet['reward']))
        campfire = dict(zip(*packet['campfire']))
        for j, i in enumerate(rows):
            probability = packet['probabilities'][j].astype(np.float64)
            probability /= probability.sum()
            action = min(len(probability)-1, int(np.searchsorted(
                probability.cumsum(), draws[positions[i], i, 0], side='right')))
            actions[i] = action
            estimates[i] = float(packet['values'][j]), float(packet['logprobs'][j, action])
            upper_policy[i] = reward.get(j), campfire.get(j)
            del upper_waiting[i]
        stats['upper_rows'] += len(rows)
        stats['upper_batches'] += 1
        upper_pending = None

    try:
        stream = Stream(model, vec, rollout_buffer, positions, seeds)
        callback.on_rollout_start()
        while not rollout_buffer.full:
            tick = time.monotonic()
            calls = {}
            for i in sorted(ready):
                mask = np.asarray(mask_getters[i](), bool)
                masks[i] = mask
                legal = (int(model._last_obs['phase'][i, 0]) == combat
                         and not model._last_obs['combat_exit_anchor'][i].any())
                legal_count = np.count_nonzero(mask)
                if legal and legal_count == 1:
                    actions[i] = int(np.flatnonzero(mask)[0])
                    estimates[i] = np.nan, 0.
                    upper_policy[i] = None, None
                    stats['forced_rows'] += 1
                elif legal and legal_count > 1 and draws[positions[i], i, 1] < config.fraction:
                    # Runic Dome has no public current intent: retain model behavior.
                    if not np.any(model._last_obs['relics'][i] == runic_dome):
                        calls[i] = mask
                    else:
                        upper_waiting[i] = time.monotonic()
                else:
                    upper_waiting[i] = time.monotonic()
                stream.stats['eligible_roots'] += int(legal and legal_count > 1)
            ready.clear()
            stats['root_prepare_seconds'] += time.monotonic()-tick
            tick = time.monotonic()
            stream.submit(calls)
            stats['root_submit_seconds'] += time.monotonic()-tick
            consume_upper()
            if upper_pending is None and upper_waiting and (
                    len(upper_waiting) >= 64 or not stream.pending
                    or time.monotonic()-min(upper_waiting.values()) >= .002):
                tick = time.monotonic()
                rows = sorted(upper_waiting)[:64]
                obs = {k: v[rows] for k, v in model._last_obs.items()}
                row_masks = np.stack([masks[i] for i in rows])
                upper_pending = (stream.service.begin_call('upper',
                    (obs, row_masks, np.asarray(rows), vec.num_envs)), rows)
                stats['policy_seconds'] += time.monotonic()-tick
            consume_upper()
            tick = time.monotonic()
            decisions = stream.receive(timeout=.001 if len(actions) < 32 and stream.pending else 0.)
            stats['search_wait_seconds'] += time.monotonic()-tick
            for i, result in decisions.items():
                actions[i] = result['action']
                estimates[i] = result['value'], result['log_prob']
                upper_policy[i] = result['reward'], result['campfire']
            consume_upper()
            if not actions:
                action_deadline = None
                continue
            # Ready slots remain exclusively owned here. Coalesce arrivals at
            # the existing step boundary, without a vector-wide search barrier.
            # Seeds/draws belong to (position, slot), never completion order.
            # A bounded wait prevents a slow native fallback from holding up
            # unrelated games or the last partially completed rollout slots.
            if action_deadline is None:
                action_deadline = time.monotonic() + .004
            if (len(actions) < 32 and (stream.pending or upper_pending is not None)
                    and time.monotonic() < action_deadline):
                continue
            action_deadline = None
            tick = time.monotonic()
            indices = sorted(actions)
            ordered = {i: actions.pop(i) for i in indices}
            rows = np.asarray(indices, np.intp)
            # Full observations are required by the current critic/learner.
            observation_copy.copy(positions[rows]*vec.num_envs+rows, rows)
            monitor_keys = {'reward_keys'} | {key for key,_,_ in reward_monitor.KINDS.values()}
            old = {k: model._last_obs[k][rows] for k in monitor_keys}
            action_rows = np.asarray(list(ordered.values()), np.int64)
            mask_rows = np.stack([masks.pop(i) for i in indices])
            value_rows, logp_rows = np.asarray([estimates.pop(i) for i in indices], np.float32).T
            episode_starts = model._last_episode_starts[rows].copy()
            for j, i in enumerate(indices):
                if np.isnan(value_rows[j]):
                    deferred_values.append((int(positions[i]), i))
            stream.before(ordered)
            observations, rewards, dones, infos = vec._step_subset(ordered, borrow=True)
            env.episode_returns[rows] += rewards
            env.episode_lengths[rows] += 1
            for j, i in enumerate(indices):
                if dones[j]:
                    episode = dict(r=float(env.episode_returns[i]), l=int(env.episode_lengths[i]),
                                   t=round(time.time()-env.t_start, 6))
                    for key in env.info_keywords:
                        episode[key] = infos[j][key]
                    infos[j]['episode'] = episode
                    if env.results_writer is not None:
                        env.results_writer.write_row(episode)
                    env.episode_returns[i] = 0.
                    env.episode_lengths[i] = 0
                    env.episode_count += 1
                if dones[j] and infos[j].get('TimeLimit.truncated') and 'terminal_observation' in infos[j]:
                    terminal = {k: v[None] for k, v in infos[j]['terminal_observation'].items()}
                    # Only the value is consumed; the terminal legal mask may be empty.
                    placeholder = np.ones((1, vec.action_space.n), bool)
                    bootstrap = stream.service.call('values', (terminal, placeholder, np.array([0])))
                    rewards[j] += model.gamma*bootstrap[0]
            reward_rows, campfire_rows = [], []
            for j, i in enumerate(indices):
                rp, cp = upper_policy.pop(i)
                if rp is not None: reward_rows.append((j, rp))
                if cp is not None: campfire_rows.append((j, cp))
            reward_monitor.record_step(model, old, action_rows, mask_rows,
                ([r[0] for r in reward_rows], np.asarray([r[1] for r in reward_rows])), infos)
            campfire_monitor.record_step(model, action_rows, mask_rows,
                ([r[0] for r in campfire_rows], np.asarray([r[1] for r in campfire_rows])), infos)
            rollout_buffer.add_subset(rows, None, action_rows, rewards, episode_starts,
                                      value_rows, logp_rows, action_masks=mask_rows)
            model.num_timesteps += len(rows)
            model._update_info_buffer(infos, dones)
            model._last_episode_starts[rows] = dones
            ready.update(i for i in indices if positions[i] < n_rollout_steps)
            prefix = sum(info.get('simulator_transition') is False for info in infos)
            stats['selection_prefix_decisions'] += prefix
            stats['simulator_transitions'] += len(rows)-prefix
            stats['transitions'] += len(rows)
            stats['transition_batches'] += 1
            stats['transitions_with_upper_pending'] += len(rows) if upper_pending is not None else 0
            stats['transitions_with_fallback_pending'] += len(rows) if stream.fallback is not None else 0
            stats['env_step_seconds'] += time.monotonic()-tick
            n_steps = int(positions.min())
            callback.update_locals(locals())
            if not callback.on_step():
                return False
        if stream.pending or upper_waiting or upper_pending is not None or actions or ready:
            raise RuntimeError('full fixed rollout retained pending work')
        tick = time.monotonic()
        # Old forced roots bypass model selection. The current learner still
        # consumes their frozen critic, batched once before GAE (never zeroed).
        for start in range(0, len(deferred_values), 512):
            coords = np.asarray(deferred_values[start:start+512], np.intp)
            t, e = coords.T
            obs = {k: v[t,e] for k, v in rollout_buffer.observations.items()}
            answer = stream.service.call('values',
                (obs, rollout_buffer.action_masks[t,e], np.arange(len(e))))
            rollout_buffer.values[t,e] = answer
        values = stream.service.call('values', (model._last_obs,
            np.stack([get_mask() for get_mask in mask_getters]), np.arange(vec.num_envs)))
        stats['value_rows'] = len(deferred_values)+vec.num_envs
        stats['value_seconds'] = time.monotonic()-tick
        if not np.isfinite(rollout_buffer.values).all():
            raise FloatingPointError('incomplete frozen critic values')
        stream.finish()
        rollout_buffer.compute_returns_and_advantage(torch.from_numpy(values), model._last_episode_starts)
        apply_room_returns(model, rollout_buffer)
        model.deck_search_rng = rng.bit_generator.state
        stats['collection_seconds'] = time.monotonic()-started
        stats.update({'vector_'+key+'_seconds': value-vector_start[key]
                      for key,value in vec.step_seconds.items()})
        stats['effective_nonforced_per_second'] = stream.stats['policy_rows']/stats['collection_seconds']
        # Keep integer work and local timing before the distributed scalar
        # reducer averages ranks. This is diagnostic output, never model input.
        import json
        (stream.snapshot.parent/f'legacy-collection-{model.num_timesteps}.json').write_text(
            json.dumps(dict(collection=stats, search=stream.stats, owner=stream.owner_report),
                       sort_keys=True), encoding='utf-8')
        model._performance_rollout = dict(policy=stats['policy_seconds']+stats['value_seconds'],
            env=stats['env_step_seconds'], mask=0., total=stats['collection_seconds'],
            overhead=max(0., stats['collection_seconds']-stats['policy_seconds']
                         -stats['value_seconds']-stats['env_step_seconds']),
            transitions=stats['transitions'],
            calls=dict(policy=stats['upper_batches'], env=stats['transition_batches'], mask=0))
        for key, value in stats.items():
            model.logger.record('pure_rl/legacy_'+key, value)
        callback.update_locals(locals())
        callback.on_rollout_end()
        return True
    finally:
        for e, enabled in zip(vec.envs, previous_value_observations, strict=True):
            e.unwrapped._emit_nosl_value_observations = enabled
        if stream is not None:
            stream.close()
