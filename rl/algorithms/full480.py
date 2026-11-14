"""The executed full480 collector, with current model inputs at its boundaries."""
import time
from contextlib import nullcontext
import numpy as np
from env.core import native_actions as C, vocab as V
from rl.algorithms.independent_rollout import IndependentRolloutConfig, GameRandomStreams, _strategic_observations
from rl.algorithms.upper_ppo import UPPER_KEYS, capture_step
from rl.training import episode_rollout, reward_monitor, campfire_monitor
from rl.training.small_playout import configure_collection
from rl.search.full480.full_seeded import (Stream as SmallPlayoutStream, RemoteUpper as UpperPolicyInference,
    upper_begin, upper_take, mirror_before, mirror_after, record_start, record_end)

def collect(model, env, callback, rollout_buffer, n_rollout_steps, use_masking=True, *, config):
    """Collect real policy/search samples with weights frozen for the whole cohort.

    The same ready/submit/receive loop serves natural cohorts and fixed PPO
    horizons. Fixed rows retain their original (time, environment) positions.
    """
    from stable_baselines3.common.vec_env import VecMonitor
    from rl.training.native_vec_env import NativeSearchVecEnv
    cfg = IndependentRolloutConfig(**config)
    fixed = not getattr(model, 'episode_rollout_config', None)
    if fixed:
        from rl.algorithms.search_auxiliary import SearchRolloutBuffer, SearchTrainingConfig
        search_config = SearchTrainingConfig(**model.deck_search_config)
        if (not isinstance(rollout_buffer, SearchRolloutBuffer) or n_rollout_steps != rollout_buffer.buffer_size
                or n_rollout_steps < 1 or cfg.max_combat_turns is not None):
            raise ValueError('fixed full480 needs the existing PPO buffer and an exact per-slot horizon')
    if (not isinstance(env, VecMonitor) or type(env.venv) is not NativeSearchVecEnv
            or not use_masking
            or (not fixed and (model.nosl_is_az_combat_strategy != 'small-playout'
                               or not model.nosl_is_az_inner_only or model.nosl_is_az_replay_enabled))
            or getattr(model, '_outer_return', False) or getattr(model, '_frozen_upper', False)
            or getattr(model.policy, 'upper_auxiliary_config', None) is not None
            or getattr(model, 'amp_mode', None) not in (None, 'off', 'fp32')
            or getattr(model, 'nosl_is_az_upper_strategy', 'model') != 'model'):
        raise ValueError('independent diagnostic requires FP32 native complete-game joint small-playout')
    vec = env.venv
    vec.strategic_observation_rows_omitted = 0
    model.policy.set_training_mode(False)
    rollout_buffer.reset()
    model._upper_completed, model._upper_censored = [], 0
    model._nosl_is_az_samples = []
    model._nosl_is_az_rollout_upper_counts = {}
    model._nosl_is_az_rollout_hp_order = {}
    reward_monitor.begin_collection(model)
    campfire_monitor.begin_collection(model)
    session = episode_rollout.begin(model, env)
    started_collection = time.monotonic()
    slot_steps = np.zeros(vec.num_envs, dtype=np.int64)
    active = np.ones(vec.num_envs, dtype=bool) if fixed else vec.episode_rollout_active
    if fixed and getattr(vec, 'episode_rollout_active', None) is not None:
        raise ValueError('fixed collection cannot reuse an active natural cohort')
    record_start(session,env)
    configure_collection(model, env)
    callback.on_rollout_start()
    randoms = GameRandomStreams(cfg.seed, 0 if fixed else session['first_seed'], vec.num_envs)
    if fixed and getattr(model, '_independent_rollout_rng', None):
        randoms.load_state_dict(model._independent_rollout_rng)
    combat = V.PHASE_VOCAB_INDEX['combat']
    end_action = C.policy_offset(model.combat_action_contract)
    ready = set(range(vec.num_envs))
    upper_waiting, actions, masks, estimates = {}, {}, {}, {}
    combat_upper = set()
    upper_policy = {}
    stats = dict(root_policy_seconds=0., env_step_seconds=0., search_wait_seconds=0.,
                 upper_batches=0, upper_rows=0, transition_batches=0, simulator_transitions=0,
                 selection_prefix_decisions=0, forced_end_turn_roots=0)
    model._independent_rollout_progress = stats
    upper_pending = None
    stats.update(upper_pending_wall_seconds=0., upper_submit_cpu_seconds=0., upper_accept_cpu_seconds=0.,
                 transitions_with_upper_pending=0, transitions_with_fallback_pending=0)
    def consume_upper():
        nonlocal upper_pending
        if upper_pending is None:
            return
        ident, rows, submitted = upper_pending
        started = time.perf_counter()
        packet = upper_take(ident)
        if packet is None:
            return
        assert all(i in upper_waiting and i not in actions and i not in ready for i in rows)
        ps = packet['probabilities']
        chosen = [randoms.action(i, ps[j]) for j, i in enumerate(rows)]
        logps = packet['logprobs'][np.arange(len(rows)),chosen]
        values = packet['values']
        reward_rows = {int(j): p for j, p in zip(*packet['reward'])}
        campfire_rows = {int(j): p for j, p in zip(*packet['campfire'])}
        for j, i in enumerate(rows):
            del upper_waiting[i]
            combat_upper.discard(i)
            actions[i] = chosen[j]
            estimates[i] = (float(values[j]), float(logps[j]))
            upper_policy[i] = (reward_rows.get(j), campfire_rows.get(j))
        stats['upper_accept_cpu_seconds'] += time.perf_counter() - started
        stats['upper_pending_wall_seconds'] += time.perf_counter() - submitted
        stats['upper_batches'] += 1
        stats['upper_rows'] += len(rows)
        upper_pending = None


    with (vec.omit_unused_combat_value_observations(),
          (nullcontext(vec.step_subset) if fixed else _strategic_observations(model, vec, callback)) as step_subset,
          UpperPolicyInference(model.policy.core, model.device) as upper_inference,
          SmallPlayoutStream(vec.small_playout_batch, threads=cfg.threads) as stream):
        while active.any():
            if cfg.max_combat_turns is not None:
                # Only inspect owned, ready slots before issuing another search or RNG draw.
                for i in sorted(ready):
                    if (model._last_obs['phase'][i, 0] == combat
                            and vec.exclude_overlong_combat(i, max_turns=cfg.max_combat_turns)):
                        ready.remove(i)
                if not vec.episode_rollout_active.any():
                    break
            maximum = int(slot_steps.max(initial=0)) if fixed else int(vec.episode_rollout_action_counts[active].max(initial=0))
            if ((not fixed and maximum >= session['config'].max_steps)
                    or time.monotonic()-started_collection >= (1800. if fixed else session['config'].max_seconds)):
                raise TimeoutError('independent diagnostic budget exhausted; no incomplete-game result')
            calls = {}
            for i in sorted(ready):
                mask = np.asarray(vec.envs[i].get_wrapper_attr('action_masks')(), dtype=bool)
                masks[i] = mask
                estimates[i] = (0., 0.)
                if int(model._last_obs['phase'][i, 0]) == combat:
                    if fixed and (mask.sum() == 1 or model._last_obs['combat_exit_anchor'][i].any()
                                  or randoms.search[i].random() >= search_config.fraction):
                        upper_waiting[i] = time.monotonic()
                        combat_upper.add(i)
                    elif mask[end_action] and mask.sum() == 1:
                        actions[i] = end_action
                        stats['forced_end_turn_roots'] += 1
                    elif ('Runic Dome' in (r['id'] for r in vec.native_search_batch.envs[i].thick.gc.relic_state())):
                        # This original no-search exception uses the same GPU
                        # policy queue as upper decisions, including its cache.
                        upper_waiting[i] = time.monotonic()
                        combat_upper.add(i)
                        batch = vec.small_playout_batch
                        batch.stats['dome_policy_decisions'] = batch.stats.get('dome_policy_decisions', 0) + 1
                    else:
                        calls[i] = ((), dict(state_nonce=randoms.nonce(i)))
                else:
                    upper_waiting[i] = time.monotonic()
            ready.clear()
            stream.submit(calls)
            consume_upper()
            if upper_pending is None and upper_waiting and (len(upper_waiting) >= cfg.upper_batch or not stream.pending
                    or time.monotonic() - min(upper_waiting.values()) >= cfg.upper_wait_seconds):
                started = time.perf_counter()
                rows = sorted(upper_waiting)[:cfg.upper_batch]
                obs = {k: v[rows] for k, v in model._last_obs.items()}
                combat_rows = [(j, i) for j, i in enumerate(rows) if i in combat_upper]
                if combat_rows and not fixed:
                    encoded = vec.native_root_observations.encode([vec.native_search_batch.envs[i] for _, i in combat_rows])
                    positions = [j for j, _ in combat_rows]
                    for key, value in encoded.items():
                        if key not in obs:
                            obs[key] = np.zeros((len(rows), *value.shape[1:]), dtype=value.dtype)
                        obs[key][positions] = value
                action_masks = np.stack([masks[i] for i in rows])
                upper_pending = (upper_begin(obs,action_masks),rows,started)
                stats['upper_submit_cpu_seconds'] += time.perf_counter() - started
            consume_upper()
            waiting = not actions or (cfg.mode == 'barrier'
                                      and len(actions) < int(active.sum()))
            started = time.perf_counter()
            decisions = stream.receive(timeout=.001 if waiting else 0.)
            stats['search_wait_seconds'] += time.perf_counter() - started
            actions.update({i: d['action'] for i, d in decisions.items()})
            if fixed:
                for i, decision in decisions.items():
                    estimates[i] = (decision['value'], decision['log_prob'])
                    key = (int(slot_steps[i]), i)
                    assert key not in rollout_buffer.search_actions
                    rollout_buffer.search_actions[key] = (decision['action'], decision['log_prob'])
                    if 'sample' in decision:decision['sample']['position'] = key[0]
            consume_upper()
            if not actions or (cfg.mode == 'barrier' and len(actions) < int(active.sum())):
                continue
            started = time.perf_counter()
            indices = sorted(actions)
            ordered = {i: actions[i] for i in indices}
            if upper_pending is not None:
                assert not set(upper_pending[1]).intersection(indices)
                stats['transitions_with_upper_pending'] += len(indices)
            if stream.fallback is not None and stream.fallback.pending is not None:
                stats['transitions_with_fallback_pending'] += len(indices)
            # Strategic records never consume the combat policy's encoded
            # inputs. Within a fight only phase/act/floor accrue its reward;
            # upper decisions retain every field used by the existing record.
            old_phase = model._last_obs['phase'][indices]
            old_keys = UPPER_KEYS if np.any(old_phase != combat) else {'phase', 'act', 'floor'}
            old = {k: old_phase if k == 'phase' else v[indices]
                   for k, v in model._last_obs.items() if k in old_keys}
            action_rows = np.array([actions.pop(i) for i in indices], dtype=np.int64)
            mask_rows = np.stack([masks.pop(i) for i in indices])
            value_rows, logp_rows = np.array([estimates.pop(i) for i in indices]).T
            if fixed:
                positions = slot_steps[indices]
                assert (positions < n_rollout_steps).all()
                # The original ready rows write their existing (time, env)
                # destinations directly; no second collector or staging buffer.
                for key, destination in rollout_buffer.observations.items():
                    destination[positions, indices] = model._last_obs[key][indices]
                rollout_buffer.actions[positions, indices, 0] = action_rows
                rollout_buffer.action_masks[positions, indices] = mask_rows
                rollout_buffer.episode_starts[positions, indices] = model._last_episode_starts[indices]
                rollout_buffer.values[positions, indices] = value_rows
                rollout_buffer.log_probs[positions, indices] = logp_rows
            mirror_before(ordered)
            observations, rewards, dones, infos = step_subset(ordered)
            mirror_after(ordered,infos)
            # VecMonitor has a full-vector API; preserve its per-slot accounting.
            env.episode_returns[indices] += rewards
            env.episode_lengths[indices] += 1
            for offset, i in enumerate(indices):
                if dones[offset]:
                    infos[offset]['episode'] = dict(r=float(env.episode_returns[i]),
                        l=int(env.episode_lengths[i]), t=round(time.time()-env.t_start, 6))
                    env.episode_returns[i] = 0.
                    env.episode_lengths[i] = 0
                    env.episode_count += 1
            if fixed:
                rollout_buffer.rewards[positions, indices] = rewards
                for offset, info in enumerate(infos):
                    if dones[offset] and info.get('TimeLimit.truncated'):
                        import torch
                        terminal, _ = model.policy.obs_to_tensor(info['terminal_observation'])
                        with torch.no_grad():tail = float(model.policy.predict_values(terminal).flatten()[0])
                        rollout_buffer.rewards[positions[offset], indices[offset]] += model.gamma * tail
                slot_steps[indices] += 1
                active[indices] = slot_steps[indices] < n_rollout_steps
            else:
                capture_step(model, old, action_rows, mask_rows, value_rows, logp_rows,
                             observations, dones, infos, indices=indices)
            reward_rows, campfire_rows = [], []
            for offset, i in enumerate(indices):
                rp, cp = upper_policy.pop(i, (None, None))
                if rp is not None:
                    reward_rows.append((offset, rp))
                if cp is not None:
                    campfire_rows.append((offset, cp))
            reward_monitor.record_step(model, old, action_rows, mask_rows,
                ([r[0] for r in reward_rows], np.asarray([r[1] for r in reward_rows], dtype=np.float32)), infos)
            campfire_monitor.record_step(model, action_rows, mask_rows,
                ([r[0] for r in campfire_rows], np.asarray([r[1] for r in campfire_rows], dtype=np.float32)), infos)
            record_end(session,indices,dones,infos)
            if not fixed:episode_rollout.record_terminals(session, dones, infos)
            prefix = sum(info.get('simulator_transition') is False for info in infos)
            stats['selection_prefix_decisions'] += prefix
            stats['simulator_transitions'] += len(indices)-prefix
            stats['transition_batches'] += 1
            model.num_timesteps += len(indices)
            model._update_info_buffer(infos, dones)
            for k in observations:
                model._last_obs[k][indices] = observations[k]
            model._last_episode_starts[indices] = dones
            ready.update(i for i in indices if active[i])
            stats['env_step_seconds'] += time.perf_counter()-started
            callback.update_locals(locals())
            if not callback.on_step():
                stream.cancel('callback stopped collection')
                if fixed:model._independent_rollout_rng = randoms.state_dict()
                return False
        if stream.pending or upper_waiting or actions or upper_pending is not None:
            raise RuntimeError('completed cohort retained pending work')
        stats['root_policy_seconds'] = stats['upper_submit_cpu_seconds'] + stats['upper_accept_cpu_seconds']
        stats.update(upper_inference.statistics())
        stats['combat_observation_rows_omitted'] = vec.strategic_observation_rows_omitted
    model._small_playout_stats = vec.small_playout_stats()
    model._small_playout_samples = sorted(vec.take_small_playout_samples(), key=lambda r: r['env'])
    model._upper_completed.sort(key=lambda r: (r['env'], r['episode']))
    if fixed:
        import torch
        from rl.algorithms.search_auxiliary import improved_prior
        assert (slot_steps == n_rollout_steps).all()
        rollout_buffer.pos, rollout_buffer.full = n_rollout_steps, True
        started = time.perf_counter()
        samples = model._small_playout_samples
        for offset in range(0, len(samples), 512):
            rows = samples[offset:offset+512]
            positions = [r['position'] for r in rows]
            indices = [r['env'] for r in rows]
            legal = rollout_buffer.action_masks[positions, indices].astype(bool)
            logits = np.zeros(legal.shape, np.float32)
            values = np.zeros_like(logits)
            worlds = np.empty(len(rows), np.float32)
            for j, sample in enumerate(rows):
                # Original public ordering: sorted legal codec IDs, with all
                # actions evaluated in the same complete set of worlds.
                choices = np.flatnonzero(sample['observation']['combat_legal'])
                assert np.array_equal(choices, np.flatnonzero(legal[j]))
                assert len(choices) == len(sample['action_bits'])
                assert (sample['visits'] == sample['visits'][0]).all()
                logits[j, choices] = sample['prior_logits']
                values[j, choices] = sample['values']
                worlds[j] = sample['visits'][0]
            target = improved_prior(torch.from_numpy(logits), torch.from_numpy(values),
                torch.from_numpy(legal), torch.from_numpy(worlds), search_config.target_std).numpy()
            for j, sample in enumerate(rows):
                rollout_buffer.add_search_target(sample['env'], target[j], position=sample['position'])
        stats['search_target_seconds'] = time.perf_counter()-started
        from rl.search.full480 import full_seeded
        count = full_seeded.count
        model.deck_search_last_stats = dict(searched_roots=count['roots'],
            chains=count['simulations'], atomic_steps=count['atomic_steps'],
            policy_rows=count['guide'], forced_rows=count['forced'],
            cutoff_rows=count['cuts'], executed_search_actions=len(rollout_buffer.search_actions))
        for key, value in model.deck_search_last_stats.items():model.logger.record('deck_search/'+key, value)
        from rl.algorithms.room_returns import apply_room_returns
        apply_room_returns(model, rollout_buffer)
        stats.update(fixed_steps=n_rollout_steps, real_rows=int(slot_steps.sum()))
    else:
        episode_rollout.exclude_samples(model, env)
        episode_rollout.finish(model, env, session, int(vec.episode_rollout_action_counts.max(initial=0)))
    model._independent_rollout_rng = randoms.state_dict()
    stats['collection_seconds'] = time.monotonic()-started_collection
    model._independent_rollout_diagnostic = dict(random_stream=randoms.version, **stats,
        mode=cfg.mode, note='Fixed real PPO rows.' if fixed else 'Only naturally completed games supply training targets.')
    model._nosl_is_az_rollout_timings = dict(stats)
    model._nosl_is_az_rollout_upper_counts = dict(forced_end_turn_roots=stats['forced_end_turn_roots'])
    for k, value in stats.items():
        model.logger.record('nosl_is_az/'+k, value)
    for k, value in model._small_playout_stats.items():
        model.logger.record('small_playout/'+k, value)
    callback.on_rollout_end()
    return True
