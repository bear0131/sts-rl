"""Real native transitions and CUDA allocation lifetimes for shared PPO."""
from copy import deepcopy
from multiprocessing import shared_memory

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


def test_shared_observations_masks_terminal_ownership_recovery_and_cleanup():
    serial = NativeSearchVecEnv([_environment] * 3, threads=1)
    grouped = NativeSubprocVecEnv([_environment] * 3, envs_per_worker=2, start_method='spawn')
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


def test_phase_graph_eviction_weight_refresh_and_released_scratch():
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
    graphs = SharedCollectionGraphs(core, capacity=2)
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
            for actual, snapshot in owned:
                for x, y in zip(actual, snapshot, strict=True):
                    torch.testing.assert_close(x, y, rtol=0, atol=0)
    finally:
        graphs.close()
        env.close()
    assert not graphs.entries
    with torch.no_grad(), pytest.raises(RuntimeError, match='live eval'):
        graphs(batch)
