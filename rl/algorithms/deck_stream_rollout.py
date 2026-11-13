"""Legacy ready-slot search scheduling, adapted to fixed-length PPO storage.

The old stepfull480 collector advances whichever roots have completed and
immediately resubmits their next decisions. Only the stopping/storage contract
changes here: every original slot contributes exactly n_steps real transitions.
"""
import time

import numpy as np
import torch
from stable_baselines3.common.utils import obs_as_tensor
from stable_baselines3.common.vec_env import VecMonitor
from sb3_contrib.common.maskable.utils import get_action_masks

from env.core import vocab
from rl.algorithms.deck_search import ShadowSearchCollection
from rl.algorithms.room_returns import apply_room_returns
from rl.training import reward_monitor, campfire_monitor


def collect(model, env, buffer, callback, n_rollout_steps, *, barrier=False):
    """Keep each slot's action/RNG/target order independent of GPU completion.

    barrier is a semantic diagnostic control, never a training configuration.
    Policy categorical draws and paired worlds have separate fixed (time,slot)
    coordinates, so batching cannot silently redraw an unfinished root.
    """
    from rl.training.native_subproc_vec_env import NativeSubprocVecEnv
    from rl.policies.collection_graphs import SharedCollectionGraphs
    from rl.policies.deck_latent import SearchContext
    vec = env.venv if isinstance(env, VecMonitor) else env
    if (type(vec) is not NativeSubprocVecEnv or model.device.type != 'cuda'
            or model.nosl_supervision_contract != 'shared-wholerun-ppo-room-v1'
            or model.amp_mode != 'off' or model.reasoning_updates or model.reasoning_coef
            or getattr(model, 'episode_rollout_config', None)
            or model.hierarchical_combat_hp or model.nosl_is_az
            or model.use_sde):
        raise ValueError('streaming search acting requires fixed FP32 native-subproc whole-run PPO')
    model.policy.set_training_mode(False)
    if n_rollout_steps != buffer.buffer_size:
        raise ValueError('fixed-slot rollout length must match its buffer')
    buffer.reset()
    reward_monitor.begin_collection(model)
    campfire_monitor.begin_collection(model)
    session = ShadowSearchCollection(model, env, buffer)
    n_envs = env.num_envs
    positions = buffer.slot_positions
    # Save only the generator state at a complete boundary, as before. These
    # coordinates reuse the existing checkpointed search RNG, not wall time.
    draws = session.rng.random((n_rollout_steps, n_envs, 2))
    root_seeds = session.rng.integers(0, 2**63-1, (n_rollout_steps, n_envs), dtype=np.int64)
    session.stream_positions, session.stream_seeds = positions, root_seeds
    current_masks = get_action_masks(env)
    ready, waiting, actions = set(range(n_envs)), set(), {}
    ready_at = dict.fromkeys(ready, time.monotonic())
    policy_pending = None
    estimates, monitor_rows = {}, {}
    graphs = getattr(model, '_shared_collection_graphs', None)
    if graphs is not None and not graphs.compatible(model.policy.core):
        graphs.close()
        graphs = None
    if graphs is None:
        graphs = SharedCollectionGraphs(model.policy.core)
        model._shared_collection_graphs = graphs
    graphs.begin_collection()
    model.policy.core._collection_graphs = graphs
    success = False
    stats = dict(policy_seconds=0., environment_seconds=0., receive_seconds=0.,
                 policy_batches=0, transition_batches=0, transitions=0,
                 transitions_with_search_pending=0, slot_steps_min=0, slot_steps_max=0,
                 policy_pending_seconds=0., transitions_with_policy_pending=0,
                 actor_graph_captures=0., actor_graph_hits=0., actor_graph_evictions=0., actor_graph_capture_seconds=0.,
                 actor_graph_context_rows=0., actor_graph_map_rows=0.)
    model._deck_stream_progress = stats
    model._reasoning_rollout_samples = []
    started = time.monotonic()
    try:
        session.start(hooks=False)
        callback.on_rollout_start()
        while not buffer.full:
            # Same submit/receive/ready loop as the accepted legacy collector.
            # No vector-wide search barrier and no speculative real action.
            eligible = ready - session.native_stream.blocked()
            # Legacy upper RPC batching: 64 ready rows or a 2 ms deadline;
            # searches keep ticking while the next actor packet is assembled.
            if (eligible and policy_pending is None and (len(eligible) >= 64 or not waiting
                    or time.monotonic()-min(ready_at[i] for i in eligible) >= .002)):
                before = time.monotonic()
                ids = np.asarray(sorted(eligible)[:64], dtype=np.intp)
                ready.difference_update(ids)
                for i in ids: del ready_at[i]
                obs = {k: v[ids] for k, v in model._last_obs.items()}
                masks = current_masks[ids]
                ident = session.resident_service.begin_call('actor', (obs, masks, ids, n_envs))
                policy_pending = ident, ids, obs, masks, before
                stats['policy_seconds'] += time.monotonic()-before
            if policy_pending is not None:
                before = time.monotonic()
                ident, ids, obs, masks, submitted = policy_pending
                received, packet = session.resident_service.take_reply(ident)
                if received:
                    logits, probabilities, values = packet['logprobs'], packet['probabilities'], packet['values']
                    reward_rows = dict(zip(*packet['reward']))
                    campfire_rows = dict(zip(*packet['campfire']))
                    cumulative = np.cumsum(probabilities, axis=1, dtype=np.float64)
                    uniforms = draws[positions[ids], ids, 0] * cumulative[:, -1]
                    picked = (cumulative <= uniforms[:, None]).sum(axis=1)
                    if np.any(picked >= masks.shape[1]) or not masks[np.arange(len(ids)), picked].all():
                        raise ValueError('invalid independent policy categorical sample')
                    for j, i in enumerate(ids):
                        i = int(i)
                        actions[i] = int(picked[j])
                        estimates[i] = (values[j], logits[j, picked[j]])
                        monitor_rows[i] = (reward_rows.get(j), campfire_rows.get(j))
                    legal = ((obs['phase'].reshape(-1) == vocab.PHASE_VOCAB_INDEX['combat'])
                             & ~obs['combat_exit_anchor'].any(1) & (masks.sum(1) > 1))
                    session.stats['eligible_roots'] += int(legal.sum())
                    selected = np.flatnonzero(legal & (draws[positions[ids], ids, 1] < session.config.fraction))
                    if len(selected):
                        global_ids = ids[selected]
                        context = SearchContext(*(v[selected] for v in packet['context']))
                        session.pending = (global_ids, masks[selected], torch.from_numpy(logits[selected]), context)
                        waiting.update(int(i) for i in global_ids)
                        session._resident_step(None)
                    else:
                        session.pending = None
                    policy_pending = None
                    stats['policy_pending_seconds'] += time.monotonic()-submitted
                    stats['policy_batches'] += 1
                stats['policy_seconds'] += time.monotonic()-before
            before = time.monotonic()
            session._receive_resident(.001 if (waiting and len(actions) == len(waiting)) or policy_pending else 0.)
            stats['receive_seconds'] += time.monotonic()-before
            for i in list(waiting):
                key = (int(positions[i]), i)
                if key in session.resident_items:
                    continue
                waiting.remove(i)
                chosen = buffer.search_actions.get(key)
                if chosen is not None:
                    session.stats['changed_policy_actions'] += int(actions[i] != chosen[0])
                    session.stats['executed_search_actions'] += 1
                    actions[i] = chosen[0]
                    estimates[i] = (estimates[i][0], chosen[1])
            available = sorted(actions.keys()-waiting-session.native_stream.blocked())
            if not available or (barrier and waiting):
                continue
            before = time.monotonic()
            ids = np.asarray(available, dtype=np.intp)
            ordered = {i: actions.pop(i) for i in available}
            old = {k: v[ids] for k, v in model._last_obs.items()}
            action_rows = np.asarray(list(ordered.values()), dtype=np.int64)
            mask_rows = current_masks[ids].copy()
            values, log_probs = np.asarray([estimates.pop(i) for i in available], dtype=np.float32).T
            episode_starts = model._last_episode_starts[ids].copy()
            observations, rewards, dones, infos = vec.step_subset(ordered)
            if isinstance(env, VecMonitor):
                env.episode_returns[ids] += rewards
                env.episode_lengths[ids] += 1
                for j, i in enumerate(available):
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
            reward_rows, campfire_rows = [], []
            for j, i in enumerate(available):
                rp, cp = monitor_rows.pop(i)
                if rp is not None:
                    reward_rows.append((j, rp))
                if cp is not None:
                    campfire_rows.append((j, cp))
            reward_monitor.record_step(model, old, action_rows, mask_rows,
                ([x[0] for x in reward_rows], np.asarray([x[1] for x in reward_rows])), infos)
            campfire_monitor.record_step(model, action_rows, mask_rows,
                ([x[0] for x in campfire_rows], np.asarray([x[1] for x in campfire_rows])), infos)
            # Match SB3's TimeLimit bootstrap, using the terminal observation
            # before automatic reset. Natural deaths never bootstrap.
            for j, info in enumerate(infos):
                if dones[j] and info.get('terminal_observation') is not None and info.get('TimeLimit.truncated', False):
                    with torch.no_grad():
                        terminal, _ = model.policy.obs_to_tensor(info['terminal_observation'])
                        rewards[j] += model.gamma * model.policy.predict_values(terminal)[0].item()
            buffer.add_subset(ids, old, action_rows, rewards, episode_starts,
                              values, log_probs, action_masks=mask_rows)
            model.num_timesteps += len(ids)
            model._update_info_buffer(infos, dones)
            for key in observations:
                model._last_obs[key][ids] = observations[key]
            model._last_episode_starts[ids] = dones
            current_masks[ids] = np.asarray([vec._cached_action_masks[i] for i in ids])
            for i in available:
                if positions[i] < n_rollout_steps:
                    ready.add(i); ready_at[i] = time.monotonic()
            stats['transitions'] += len(ids)
            stats['transition_batches'] += 1
            stats['transitions_with_search_pending'] += len(ids) if waiting else 0
            stats['transitions_with_policy_pending'] += len(ids) if policy_pending is not None else 0
            stats['environment_seconds'] += time.monotonic()-before
            stats['slot_steps_min'] = int(positions.min())
            stats['slot_steps_max'] = int(positions.max())
            n_steps = int(positions.min())
            callback.update_locals(locals())
            if not callback.on_step():
                return False
        if policy_pending is not None or waiting or actions or ready or session.resident_items or session.resident_service.outstanding:
            raise RuntimeError('complete rollout retained pending work')
        session.resident_service.park()
        for k, v in session.resident_service.status.get('inference', {}).get('graphs', {}).items():
            stats['actor_graph_'+k] = v
        with torch.no_grad():
            values = model.policy.predict_values(obs_as_tensor(model._last_obs, model.device))
        buffer.compute_returns_and_advantage(last_values=values, dones=model._last_episode_starts)
        session.close()  # Publish this collection's counts before return/metric construction.
        apply_room_returns(model, buffer)
        callback.update_locals(locals())
        callback.on_rollout_end()
        success = True
        return True
    finally:
        session.close()
        model.__dict__.pop('_deck_stream_progress', None)
        model.policy.core.__dict__.pop('_collection_graphs', None)
        if not success:
            graphs.close()
            model.__dict__.pop('_shared_collection_graphs', None)
        for key, value in graphs.stats.items():
            model.logger.record('pure_rl/collection_graph_'+key, value)
        for key, value in stats.items():
            model.logger.record('pure_rl/stream_'+key, value)
        model._performance_rollout = dict(policy=stats['policy_seconds'], env=stats['environment_seconds'],
            mask=0., overhead=max(0., time.monotonic()-started-stats['policy_seconds']-stats['environment_seconds']),
            total=time.monotonic()-started, transitions=stats['transitions'],
            calls=dict(policy=stats['policy_batches'], env=stats['transition_batches'], mask=0))
