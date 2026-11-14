"""Real native transitions and CUDA allocation lifetimes for shared PPO."""
from copy import deepcopy
from multiprocessing import get_all_start_methods, shared_memory

import numpy as np
import pytest
import torch
from gymnasium.wrappers import TimeLimit
from stable_baselines3.common.utils import obs_as_tensor

from env.core.run_env import SlayTheSpireEnv
from rl.policies.shared_wholerun import SharedWholeRunPolicy
from rl.algorithms.accumulating_ppo import AccumulatingMaskablePPO
from rl.training.environments import TrainingStateWrapper
from rl.training.native_vec_env import NativeSearchVecEnv
from rl.training.native_subproc_vec_env import NativeSubprocVecEnv
from rl.training.reward import StrategicRewardWrapper

pytestmark = [pytest.mark.native, pytest.mark.training, pytest.mark.timeout(180)]


def _environment():
    return TrainingStateWrapper(TimeLimit(StrategicRewardWrapper(SlayTheSpireEnv(
        card_language=True, global_map_view=True, combat_action_contract='autoregressive-v2')),
        max_episode_steps=7))


def _equal(a, b):
    if isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            _equal(a[key], b[key])
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for x, y in zip(a, b, strict=True):
            _equal(x, y)
    elif isinstance(a, np.ndarray):
        np.testing.assert_array_equal(a, b)
    else:
        assert a == b


@pytest.mark.parametrize('start_method', [m for m in ('spawn', 'forkserver')
                                        if m in get_all_start_methods()])
def test_shared_observations_masks_terminal_ownership_recovery_and_cleanup(start_method):
    serial = NativeSearchVecEnv([_environment] * 3, threads=1)
    grouped = NativeSubprocVecEnv([_environment] * 3, envs_per_worker=2, start_method=start_method)
    name = grouped.memory.name
    try:
        for env in (serial, grouped):
            env.seed(20261109)
        _equal(serial.reset(), grouped.reset())
        retained = []
        for step in range(25):
            masks = np.stack(serial.env_method('action_masks'))
            _equal(masks, np.stack(grouped.env_method('action_masks')))
            actions = [np.flatnonzero(mask)[-1] for mask in masks]
            expected, actual = serial.step(actions), grouped.step(actions)
            _equal(expected, actual)
            retained.append((actual, deepcopy(actual)))
            if step == 9:
                states = grouped.env_method('export_training_state')
                old_obs, old_masks = deepcopy(actual[0]), masks.copy()
                _equal(serial.env_method('export_training_state'), states)
                old_masks = np.stack(grouped.env_method('action_masks'))
                grouped.step([np.flatnonzero(mask)[0] for mask in old_masks])
                _equal(grouped.restore_training_states(states), old_obs)
                _equal(np.stack(grouped.env_method('action_masks')), old_masks)
        # Includes auto-reset/truncated terminal observations. Later writes to
        # shared memory must not corrupt the collector's previous observation.
        assert any(any(row[2]) for row, _ in retained)
        for actual, snapshot in retained:
            _equal(actual, snapshot)
    finally:
        serial.close()
        grouped.close()
    assert all(not process.is_alive() for process in grouped.processes)
    with pytest.raises(FileNotFoundError):
        shared_memory.SharedMemory(name=name)


@pytest.mark.parametrize('start_method', [m for m in ('spawn', 'forkserver')
                                        if m in get_all_start_methods()])
def test_shared_subset_preserves_slot_order_terminal_state_and_restored_bank(start_method):
    serial = NativeSearchVecEnv([_environment]*3, threads=1)
    grouped = NativeSubprocVecEnv([_environment]*3, envs_per_worker=2, start_method=start_method)
    retained = []
    try:
        for env in (serial, grouped):
            env.seed(20261111)
            env.reset()
        before = grouped.get_attr('_emit_nosl_value_observations')
        with serial.omit_unused_combat_value_observations(), grouped.omit_unused_combat_value_observations():
            for step in range(40):
                slots = ((2, 0), (1,), (2, 1, 0))[step % 3]
                masks = serial.env_method('action_masks', indices=slots)
                _equal(masks, grouped.env_method('action_masks', indices=slots))
                actions = {i: int(np.flatnonzero(mask)[-1]) for i, mask in zip(slots, masks, strict=True)}
                expected, actual = serial._step_subset(actions), grouped._step_subset(actions)
                _equal(expected, actual)
                _equal(serial.buf_obs, grouped.buf_obs)
                retained.append((actual, deepcopy(actual)))
        assert grouped.get_attr('_emit_nosl_value_observations') == before
        states = serial.env_method('export_training_state')
        _equal(states, grouped.env_method('export_training_state'))
        saved = deepcopy(grouped.buf_obs)
        masks = grouped.env_method('action_masks', indices=[2, 0])
        grouped._step_subset({i: int(np.flatnonzero(m)[0]) for i, m in zip((2, 0), masks)})
        _equal(grouped.restore_training_states(states), saved)
        _equal(grouped.buf_obs, saved)
        _equal(grouped.env_method('export_training_state'), states)
        for actual, snapshot in retained:
            _equal(actual, snapshot)
    finally:
        serial.close()
        grouped.close()


@pytest.mark.parametrize('capacity', [1, 2])
def test_phase_graph_eviction_weight_refresh_and_released_scratch(capacity):
    from rl.policies.collection_graphs import SharedCollectionGraphs
    assert torch.cuda.is_available(), 'explicit CUDA graph qualification required'
    torch.set_float32_matmul_precision('highest')
    env = NativeSearchVecEnv([_environment] * 4, threads=1)
    model = AccumulatingMaskablePPO(SharedWholeRunPolicy, env, seed=20261109,
        n_steps=32, batch_size=32, n_epochs=1, device='cuda', gamma=1.,
        reasoning_coef=0, reasoning_updates=0, nosl_is_az=False,
        policy_kwargs=dict(d_model=32, n_layers=1, n_heads=4))
    model.policy.set_training_mode(False)
    core = model.policy.core
    graphs = SharedCollectionGraphs(core, capacity=capacity)
    obs = env.reset()
    owned = []
    try:
        with torch.no_grad():
            for step in range(36):
                tensors = obs_as_tensor(obs, model.device)
                count = (1, 2, 4)[step % 3]
                batch = {key: value[:count] for key, value in tensors.items()}
                expected = core.actor_critic(batch)
                actual = graphs(batch)
                for x, y in zip(actual, expected, strict=True):
                    torch.testing.assert_close(x, y, rtol=2e-5, atol=2e-5)
                owned.append((actual, tuple(x.clone() for x in actual)))
                if step % 4 == 0:
                    core.combat.card.weight.add_(.0001)
                    core.combat.value_head[-1].bias.add_(.0001)
                    # Replay an existing signature after changing weights.
                    for x, y in zip(graphs(batch), core.actor_critic(batch), strict=True):
                        torch.testing.assert_close(x, y, rtol=2e-5, atol=2e-5)
                    torch._C._cuda_clearCublasWorkspaces()
                    scratch = torch.empty(8*1024*1024, device='cuda')
                    scratch.fill_(123)
                    del scratch
                    torch.cuda.empty_cache()
                masks = np.stack(env.env_method('action_masks'))
                obs, *_ = env.step([np.flatnonzero(mask)[-1] for mask in masks])
            assert graphs.stats['evictions'] > 0 and graphs.stats['hits'] > 0
            # Queue distinct inputs on alternating streams without checking
            # between calls. Shared scratch must not race another replay or a
            # caller's still-pending copy of its published output.
            variants, references, queued = [], [], []
            for index in range(4):
                variant = {key: value.clone() for key, value in batch.items()}
                variant['player_scalars'].add_(index * .1)
                variant['gold'].add_(index * .1)
                variants.append(variant)
                references.append(tuple(x.clone() for x in core.actor_critic(variant)))
            torch.cuda.synchronize()
            streams = (torch.cuda.current_stream(), torch.cuda.Stream())
            for turn, index in enumerate((3, 0, 2, 1, 0, 3, 1, 2) * 2):
                with torch.cuda.stream(streams[turn % 2]):
                    result = graphs(variants[index])
                    queued.append((result, references[index]))
                    scratch = torch.empty(1024 * 1024, device='cuda')
                    scratch.fill_(turn)
                    del scratch
            graphs.close()
            torch.cuda.empty_cache()
            for actual, expected in queued:
                for x, y in zip(actual, expected, strict=True):
                    torch.testing.assert_close(x, y, rtol=2e-5, atol=2e-5)
            for actual, snapshot in owned:
                for x, y in zip(actual, snapshot, strict=True):
                    torch.testing.assert_close(x, y, rtol=0, atol=0)
    finally:
        graphs.close()
        env.close()
    assert not graphs.entries
    with torch.no_grad(), pytest.raises(RuntimeError, match='live eval'):
        graphs(batch)


@pytest.mark.parametrize('start_method', [m for m in ('spawn', 'forkserver')
                                        if m in get_all_start_methods()])
def test_native_transport_reports_worker_death_and_releases_shared_memory(start_method):
    env = NativeSubprocVecEnv([_environment] * 3, envs_per_worker=2, start_method=start_method)
    name = env.memory.name
    try:
        env.reset()
        masks = env.env_method('action_masks')
        env.step_async(np.asarray([np.flatnonzero(mask)[0] for mask in masks]))
        env.processes[1].terminate()
        env.processes[1].join(timeout=5)
        with pytest.raises(RuntimeError, match=r'group=1 logical_envs=\[2\]'):
            env.step_wait()
    finally:
        env.close()
    assert all(not process.is_alive() for process in env.processes)
    with pytest.raises(FileNotFoundError):
        shared_memory.SharedMemory(name=name)


@pytest.mark.parametrize('capacity', [1, 4])
@pytest.mark.parametrize('host_metadata', [False, True])
def test_map_cache_tracks_content_weights_shape_and_schema(capacity, host_metadata):
    from env.core import vocab
    from rl.policies.collection_graphs import SharedCollectionGraphs
    assert torch.cuda.is_available(), 'explicit CUDA map-cache qualification required'
    torch.set_float32_matmul_precision('highest')
    env = SlayTheSpireEnv(card_language=True, global_map_view=True, upper_public_context=True,
                         event_option_observation=True, neow_observation=True,
                         combat_action_contract='autoregressive-v2')
    graphs = None
    try:
        frame, _ = env.reset(seed=20261109)
        frames = {}
        combat = vocab.PHASE_VOCAB_INDEX['combat']
        for _ in range(128):
            frames.setdefault(int(frame['phase'][0]), deepcopy(frame))
            if combat in frames and len(frames) >= 3:
                break
            frame, _, ended, truncated, _ = env.step(int(np.flatnonzero(env.action_masks())[0]))
            if ended or truncated:
                frame, _ = env.reset()
        assert combat in frames and len(frames) >= 3
        model = AccumulatingMaskablePPO(SharedWholeRunPolicy, env, seed=20261109,
            n_steps=4, batch_size=4, n_epochs=1, device='cuda', gamma=1.,
            reasoning_coef=0, reasoning_updates=0, nosl_is_az=False,
            policy_kwargs=dict(d_model=32, n_layers=1, n_heads=4))
        model.policy.set_training_mode(False)
        core = model.policy.core
        graphs = SharedCollectionGraphs(core, capacity=capacity)
        invocations = 0
        def forward(observation):
            nonlocal invocations
            invocations += 1
            if host_metadata:
                graphs.bind_host_observations(observation,
                    {key: value.detach().cpu().numpy() for key, value in observation.items()})
            # Owned graph buffers must also survive mode changes, including a
            # subsequent no_grad bootstrap after an inference-mode collection.
            with torch.inference_mode(host_metadata and bool(invocations % 2)), torch.no_grad():
                return graphs(observation)
        rows = list(frames.values())
        with torch.no_grad():
            for count in (7, 33, 65, 129, 193, 225, 7):
                chosen = [rows[(i * 2 + 1) % len(rows)] for i in range(count)]
                obs = obs_as_tensor({k: np.stack([r[k] for r in chosen]) for k in chosen[0]},
                                    model.device)
                for actual, expected in zip(forward(obs), core.actor_critic(obs), strict=True):
                    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)
                processed = graphs.stats['map_rows']
                forward(obs)
                assert graphs.stats['map_rows'] == processed
                if host_metadata:
                    graphs.bind_host_observations(obs,
                        {key: value.detach().cpu().numpy() for key, value in obs.items()})
                    # A binding for a different dict cannot be reused by an
                    # arbitrary caller, even when it contains the same tensors.
                    other = dict(obs)
                    for actual, expected in zip(graphs(other), core.actor_critic(other), strict=True):
                        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)
                forward(obs)
                processed = graphs.stats['map_rows']
                upper_row = int((obs['phase'].flatten() != combat).nonzero()[0])
                node = int(obs['map_full_rooms'][upper_row].ne(0).nonzero()[0])
                obs['map_full_features'][upper_row, node, 2] = 1 - obs['map_full_features'][upper_row, node, 2]
                for actual, expected in zip(forward(obs), core.actor_critic(obs), strict=True):
                    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)
                assert graphs.stats['map_rows'] == processed + 1
                # The schema itself participates in the graph identity; a
                # reordered dict cannot reinterpret packed field offsets.
                obs = dict(reversed(list(obs.items())))
                core.overworld.encoder.map_dag.node[0].bias.add_(.01)
                core.combat.card.weight.add_(.001)
                graphs.begin_collection()
                torch._C._cuda_clearCublasWorkspaces()
                scratch = torch.empty(1024 * 1024, device='cuda')
                scratch.fill_(17)
                del scratch
                torch.cuda.empty_cache()
                for actual, expected in zip(forward(obs), core.actor_critic(obs), strict=True):
                    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)
                assert graphs.stats['map_rows'] > 0
                assert len(graphs.entries) <= capacity and len(graphs.map_entries) <= capacity
    finally:
        if graphs is not None:
            graphs.close()
        env.close()
