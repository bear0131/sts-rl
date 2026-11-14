def collect(model, env, callback, rollout_buffer, n_rollout_steps, use_masking=True, *, config):
    """Collect real policy/search samples with weights frozen for the whole cohort.

    An explicit synchronous control isolates scheduling from the new RNG
    contract. Per-game sample order is canonicalized before returning.
    Legacy unused combat-value diagnostic rows are not materialized in either
    mode. Complete cohorts are the only supported checkpoint boundary.
    """
    from stable_baselines3.common.vec_env import VecMonitor
    from rl.training.native_vec_env import NativeSearchVecEnv
    cfg = IndependentRolloutConfig(**config)
    if (not isinstance(env, VecMonitor) or type(env.venv) is not NativeSearchVecEnv
            or not use_masking or not getattr(model, 'episode_rollout_config', None)
            or model.nosl_is_az_combat_strategy != 'small-playout'
            or not model.nosl_is_az_inner_only or model.nosl_is_az_replay_enabled
            or getattr(model, '_outer_return', False) or getattr(model, '_frozen_upper', False)
            or model.policy.upper_auxiliary_config is not None
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
    record_start(session,env)
    configure_collection(model, env)
    callback.on_rollout_start()
    randoms = GameRandomStreams(cfg.seed, session['first_seed'], vec.num_envs)
    combat = V.PHASE_VOCAB_INDEX['combat']
    end_action = C.policy_offset(model.combat_action_contract)
    ready = set(range(vec.num_envs))
    upper_waiting, actions, masks, estimates = {}, {}, {}, {}
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
            actions[i] = chosen[j]
            estimates[i] = (float(values[j]), float(logps[j]))
            upper_policy[i] = (reward_rows.get(j), campfire_rows.get(j))
        stats['upper_accept_cpu_seconds'] += time.perf_counter() - started
        stats['upper_pending_wall_seconds'] += time.perf_counter() - submitted
        stats['upper_batches'] += 1
        stats['upper_rows'] += len(rows)
        upper_pending = None


    with (vec.omit_unused_combat_value_observations(),
          _strategic_observations(model, vec, callback) as step_subset,
          UpperPolicyInference(model.policy.core, model.device) as upper_inference,
          SmallPlayoutStream(vec.small_playout_batch, threads=cfg.threads) as stream):
        while vec.episode_rollout_active.any():
            if cfg.max_combat_turns is not None:
                # Only inspect owned, ready slots before issuing another search or RNG draw.
                for i in sorted(ready):
                    if (model._last_obs['phase'][i, 0] == combat
                            and vec.exclude_overlong_combat(i, max_turns=cfg.max_combat_turns)):
                        ready.remove(i)
                if not vec.episode_rollout_active.any():
                    break
            maximum = int(vec.episode_rollout_action_counts[vec.episode_rollout_active].max(initial=0))
            if (maximum >= session['config'].max_steps
                    or time.monotonic()-session['started'] >= session['config'].max_seconds):
                raise TimeoutError('independent diagnostic budget exhausted; no incomplete-game result')
            calls = {}
            for i in sorted(ready):
                mask = np.asarray(vec.envs[i].get_wrapper_attr('action_masks')(), dtype=bool)
                masks[i] = mask
                estimates[i] = (0., 0.)
                if int(model._last_obs['phase'][i, 0]) == combat:
                    if mask[end_action] and mask.sum() == 1:
                        actions[i] = end_action
                        stats['forced_end_turn_roots'] += 1
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
                action_masks = np.stack([masks[i] for i in rows])
                upper_pending = (upper_begin(obs,action_masks),rows,started)
                stats['upper_submit_cpu_seconds'] += time.perf_counter() - started
            consume_upper()
            waiting = not actions or (cfg.mode == 'barrier'
                                      and len(actions) < int(vec.episode_rollout_active.sum()))
            started = time.perf_counter()
            decisions = stream.receive(timeout=.001 if waiting else 0.)
            stats['search_wait_seconds'] += time.perf_counter() - started
            actions.update({i: d['action'] for i, d in decisions.items()})
            consume_upper()
            if not actions or (cfg.mode == 'barrier' and len(actions) < int(vec.episode_rollout_active.sum())):
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
            episode_rollout.record_terminals(session, dones, infos)
            prefix = sum(info.get('simulator_transition') is False for info in infos)
            stats['selection_prefix_decisions'] += prefix
            stats['simulator_transitions'] += len(indices)-prefix
            stats['transition_batches'] += 1
            model.num_timesteps += len(indices)
            model._update_info_buffer(infos, dones)
            for k in observations:
                model._last_obs[k][indices] = observations[k]
            model._last_episode_starts[indices] = dones
            ready.update(i for i in indices if vec.episode_rollout_active[i])
            stats['env_step_seconds'] += time.perf_counter()-started
            callback.update_locals(locals())
            if not callback.on_step():
                return False
        if stream.pending or upper_waiting or actions or upper_pending is not None:
            raise RuntimeError('completed cohort retained pending work')
        stats['root_policy_seconds'] = stats['upper_submit_cpu_seconds'] + stats['upper_accept_cpu_seconds']
        stats.update(upper_inference.statistics())
        stats['combat_observation_rows_omitted'] = vec.strategic_observation_rows_omitted
    model._small_playout_stats = vec.small_playout_stats()
    model._small_playout_samples = sorted(vec.take_small_playout_samples(), key=lambda r: r['env'])
    model._upper_completed.sort(key=lambda r: (r['env'], r['episode']))
    episode_rollout.exclude_samples(model, env)
    episode_rollout.finish(model, env, session, int(vec.episode_rollout_action_counts.max(initial=0)))
    model._independent_rollout_rng = randoms.state_dict()
    stats['collection_seconds'] = time.monotonic()-session['started']
    model._independent_rollout_diagnostic = dict(random_stream=randoms.version, **stats,
        mode=cfg.mode, note='Only naturally completed games supply training targets.')
    model._nosl_is_az_rollout_timings = dict(stats)
    model._nosl_is_az_rollout_upper_counts = dict(forced_end_turn_roots=stats['forced_end_turn_roots'])
    for k, value in stats.items():
        model.logger.record('nosl_is_az/'+k, value)
    for k, value in model._small_playout_stats.items():
        model.logger.record('small_playout/'+k, value)
    callback.on_rollout_end()
    return True
