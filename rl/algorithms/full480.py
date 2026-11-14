"""The original executed full480 collector, adapted at its consumer boundaries."""
import time
import numpy as np
import torch
from env.core import vocab as V
from rl.algorithms.independent_rollout import IndependentRolloutConfig, GameRandomStreams
from rl.algorithms.search_auxiliary import SearchTrainingConfig, SearchRolloutBuffer
from rl.algorithms.room_returns import apply_room_returns
from rl.training import reward_monitor, campfire_monitor
from rl.search.full480.full_seeded import Stream as SmallPlayoutStream, upper_begin, upper_take, mirror_before, mirror_after


def collect(model, env, callback, rollout_buffer, n_rollout_steps, use_masking=True, *, config):
    """Drain original async search/upper queues into fixed per-slot PPO rows."""
    from stable_baselines3.common.vec_env import VecMonitor
    from rl.training.native_vec_env import NativeSearchVecEnv
    search = SearchTrainingConfig(**config)
    cfg = IndependentRolloutConfig(seed=search.seed, threads=8, upper_batch=64, upper_wait_seconds=0.)
    if (not isinstance(env, VecMonitor) or type(env.venv) is not NativeSearchVecEnv
            or not use_masking or getattr(model, 'episode_rollout_config', None)
            or not isinstance(rollout_buffer, SearchRolloutBuffer)
            or model.nosl_supervision_contract != 'shared-wholerun-ppo-room-v1'
            or model.policy.core.model_contract != 'deck-latent-combat-v1'
            or model.policy.combat_value_target != 'pre-reward-anchor'
            or search.action_selection != 'search-filter-policy-rank-v1'
            or search.max_depth > 200
            or getattr(model, 'amp_mode', None) not in (None, 'off', 'fp32')
            or n_rollout_steps != rollout_buffer.buffer_size):
        raise ValueError('full480 requires native FP32 Deck PPO with fixed row storage and search-led actions')
    vec = env.venv
    vec.strategic_observation_rows_omitted = 0
    model.policy.set_training_mode(False)
    rollout_buffer.reset()
    reward_monitor.begin_collection(model)
    campfire_monitor.begin_collection(model)
    started_collection = time.monotonic()
    callback.on_rollout_start()
    randoms = GameRandomStreams(cfg.seed, 0, vec.num_envs)
    from rl.search.full480.finite_plan import WORLD_STATE_VERSION
    model.full480_search_rng_version = WORLD_STATE_VERSION
    if getattr(model, '_full480_rng', None) is not None:
        randoms.load_state_dict(model._full480_rng)
    combat = V.PHASE_VOCAB_INDEX['combat']
    ready = set(range(vec.num_envs))
    upper_waiting, actions, masks, estimates = {}, {}, {}, {}
    upper_policy = {}
    search_targets, search_actions = {}, {}
    stats = dict(root_policy_seconds=0., env_step_seconds=0., search_wait_seconds=0.,
                 upper_batches=0, upper_rows=0, transition_batches=0, simulator_transitions=0,
                 selection_prefix_decisions=0, forced_end_turn_roots=0,
                 eligible_roots=0, selected_roots=0, changed_policy_actions=0)
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
        calls = {}
        for j, i in enumerate(rows):
            del upper_waiting[i]
            estimates[i] = (float(values[j]), float(logps[j]))
            upper_policy[i] = (reward_rows.get(j), campfire_rows.get(j))
            live_combat = (int(model._last_obs['phase'][i, 0]) == combat
                           and not model._last_obs['combat_exit_anchor'][i].any())
            eligible = live_combat and int(masks[i].sum()) > 1
            if eligible:
                eligible = not any(r['id']=='Runic Dome' for r in vec.native_search_batch.envs[i].thick.gc.relic_state())
            stats['eligible_roots'] += int(eligible)
            if eligible and randoms.search[i].random() < search.fraction:
                calls[i] = dict(nonce=randoms.nonce(i),observation={k:v[i].copy() for k,v in model._last_obs.items()},
                    mask=masks[i],logits=packet['logits'][j],context=tuple(v[j] for v in packet['context']),policy_action=chosen[j])
                stats['selected_roots'] += 1
            else:
                actions[i] = chosen[j]
                stats['forced_end_turn_roots'] += int(live_combat and masks[i].sum()==1)
        stream.submit(calls)
        stats['upper_accept_cpu_seconds'] += time.perf_counter() - started
        stats['upper_pending_wall_seconds'] += time.perf_counter() - submitted
        stats['upper_batches'] += 1
        stats['upper_rows'] += len(rows)
        upper_pending = None


    with SmallPlayoutStream(vec.native_search_batch,threads=cfg.threads,model=model,config=search) as stream:
        step_subset=vec.step_subset
        while not rollout_buffer.full:
            if time.monotonic()-started_collection > search.collection_timeout_seconds:
                raise TimeoutError('full480 collection deadline exhausted; partial PPO buffer is not accepted')
            for i in sorted(ready):
                mask = np.asarray(vec.envs[i].get_wrapper_attr('action_masks')(), dtype=bool)
                masks[i] = mask
                upper_waiting[i] = time.monotonic()
            ready.clear()
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
            waiting = not actions
            started = time.perf_counter()
            decisions = stream.receive(timeout=.001 if waiting else 0.)
            stats['search_wait_seconds'] += time.perf_counter() - started
            actions.update({i: d['action'] for i, d in decisions.items()})
            for i,decision in decisions.items():
                value,_=estimates[i];estimates[i]=(value,decision['log_prob'])
                search_targets[i]=decision['target'];search_actions[i]=(decision['action'],decision['log_prob'])
                stats['changed_policy_actions'] += decision['changed']
            consume_upper()
            if not actions:
                continue
            started = time.perf_counter()
            indices = sorted(actions)
            ordered = {i: actions[i] for i in indices}
            if upper_pending is not None:
                assert not set(upper_pending[1]).intersection(indices)
                stats['transitions_with_upper_pending'] += len(indices)
            if stream.fallback is not None and stream.fallback.pending is not None:
                stats['transitions_with_fallback_pending'] += len(indices)
            # Current PPO consumes every real pre-action field, including combat.
            old = {k:v[indices] for k,v in model._last_obs.items()}
            episode_starts = model._last_episode_starts[indices].copy()
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
            for offset,info in enumerate(infos):
                if dones[offset] and info.get('TimeLimit.truncated',False):
                    terminal={k:v[None] for k,v in info['terminal_observation'].items()}
                    bootstrap=stream.service.call('upper',(terminal,np.ones((1,mask_rows.shape[1]),bool)))['values'][0]
                    rewards[offset]+=model.gamma*float(bootstrap)
            rollout_buffer.add_rows(np.asarray(indices,np.int64),old,action_rows,rewards,episode_starts,value_rows,logp_rows,
                action_masks=mask_rows,search_targets={i:search_targets.pop(i) for i in indices if i in search_targets},
                search_actions={i:search_actions.pop(i) for i in indices if i in search_actions})
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
            prefix = sum(info.get('simulator_transition') is False for info in infos)
            stats['selection_prefix_decisions'] += prefix
            stats['simulator_transitions'] += len(indices)-prefix
            stats['transition_batches'] += 1
            model.num_timesteps += len(indices)
            model._update_info_buffer(infos, dones)
            for k in observations:
                model._last_obs[k][indices] = observations[k]
            model._last_episode_starts[indices] = dones
            positions=rollout_buffer.row_positions
            ready.update(i for i in indices if positions[i]<n_rollout_steps)
            stats['env_step_seconds'] += time.perf_counter()-started
            callback.update_locals(locals())
            if not callback.on_step():
                model._full480_rng = randoms.state_dict()
                stream.cancel('callback stopped collection')
                return False
        if stream.pending or upper_waiting or actions or upper_pending is not None:
            raise RuntimeError('completed fixed rollout retained pending work')
        stats['root_policy_seconds'] = stats['upper_submit_cpu_seconds'] + stats['upper_accept_cpu_seconds']
        stats.update(stream.service.call('upper_statistics'))
        masks=np.stack([e.action_masks() for e in vec.native_search_batch.envs])
        last_values=stream.service.call('upper',(model._last_obs,masks))['values']
        from rl.search.full480.full_seeded import count
        stats.update(chains=count['simulations'],atomic_steps=count['atomic_steps'],policy_rows=count['guide'],
            forced_rows=count['forced'],anchor_rows=count['anchors'],cutoff_rows=count['cuts'],
            searched_roots=count['roots'],executed_search_actions=count['roots'])
    model._full480_rng = randoms.state_dict()
    model._reasoning_rollout_samples = []
    rollout_buffer.compute_returns_and_advantage(torch.as_tensor(last_values),model._last_episode_starts)
    model.deck_search_last_stats=dict(stats)
    apply_room_returns(model,rollout_buffer,last_values=torch.as_tensor(last_values))
    stats['collection_seconds'] = time.monotonic()-started_collection
    model.deck_search_last_stats=dict(stats)
    from rl.search.full480.full_seeded import save,report
    save('complete',report())
    model._independent_rollout_diagnostic = dict(random_stream=randoms.version,
        search_random_stream=WORLD_STATE_VERSION, **stats,
        mode=cfg.mode, note='Fixed real PPO rows; complete paired search targets; original full480 scheduling.')
    model._nosl_is_az_rollout_timings = dict(stats)
    for k, value in stats.items():
        model.logger.record('deck_search/'+k, value)
    callback.on_rollout_end()
    return True
