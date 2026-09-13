"""A real replay batch follows the same learner/Adam path through local and ring adapters."""
import copy
import numpy as np
import pytest
import torch

from rl.algorithms.accumulating_ppo import AccumulatingMaskablePPO
from rl.training.distributed_sb3 import distributed_nosl_is_az_update
from tests.integration.test_training_smoke import train_once, assert_nested_equal

pytestmark = [pytest.mark.native, pytest.mark.training]


def test_single_rank_ring_matches_local_replay_optimizer(tmp_path, short_combat_env_type):
    model, env = train_once(True, env_type=short_combat_env_type,
        nosl_is_az=True, nosl_is_az_cpp=True, nosl_is_az_inner_only=True,
        nosl_is_az_depth=0, nosl_is_az_simulations=4,
        nosl_is_az_search_mode='full_gumbel', nosl_is_az_paper_full_gumbel=True,
        nosl_is_az_replay_capacity=128, nosl_is_az_micro_batch_size=8,
        nosl_is_az_leaf_batch_size=1)
    try:
        model._prepare_nosl_is_az_replay()
        model.nosl_is_az_updates = 2
        path = tmp_path / 'shared.zip'
        model.save(path)
        ring_model = AccumulatingMaskablePPO.load(path, device='cpu')
        for name in ('_nosl_is_az_replay', '_nosl_is_az_replay_prepare_stats'):
            setattr(ring_model, name, copy.deepcopy(getattr(model, name)))
        model.policy.set_training_mode(True)
        ring_model.policy.set_training_mode(True)
        model.configure_nosl_is_az_inner_only()
        ring_model.configure_nosl_is_az_inner_only()

        class LoopbackRing:
            def average(self, tensor):
                return tensor.clone(), 0

        numpy_rng, torch_rng = np.random.get_state(), torch.get_rng_state()
        local = model._train_nosl_is_az(prepare_replay=False)
        np.random.set_state(numpy_rng)
        torch.set_rng_state(torch_rng)
        ring = distributed_nosl_is_az_update(ring_model, rank=0, world_size=1, peers=[],
            key=tmp_path/'unused-key', control_dir=tmp_path, local_root=tmp_path,
            remote_root=str(tmp_path), run_id='loopback', update=0,
            ring_transport=LoopbackRing(), barrier_timeout=2, prepare_replay=False)
        assert local['updates'] == ring['updates'] == 2
        assert_nested_equal(model.policy.state_dict(), ring_model.policy.state_dict())
        assert_nested_equal(model.policy.optimizer.state_dict(), ring_model.policy.optimizer.state_dict())
    finally:
        env.close()
