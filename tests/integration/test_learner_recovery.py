"""Recovery keeps completed replay, Adam and sampling RNG at the same boundary."""
import copy
import random

import numpy as np
import pytest
import torch

from rl.algorithms.accumulating_ppo import AccumulatingMaskablePPO
from rl.training.checkpoints import read_bundle, restore_learner, save_bundle
from tests.integration.test_training_smoke import train_once, assert_nested_equal

pytestmark = [pytest.mark.native, pytest.mark.training]


def test_replay_rng_and_next_update_match_after_recovery(tmp_path, short_combat_env_type):
    model, env = train_once(True, env_type=short_combat_env_type,
        nosl_is_az=True, nosl_is_az_cpp=True, nosl_is_az_inner_only=True,
        nosl_is_az_simulations=4, nosl_is_az_replay_capacity=128,
        nosl_is_az_micro_batch_size=8, nosl_is_az_leaf_batch_size=1)
    try:
        model._prepare_nosl_is_az_replay()
        model.configure_nosl_is_az_inner_only()
        model.nosl_is_az_updates = 1
        # Simulate a process killed during the next generation's write, before
        # manifest publication; recovery must not get stuck at this step.
        interrupted = tmp_path / 'rank0' / f'step-{model.num_timesteps:012d}'
        interrupted.mkdir(parents=True)
        (interrupted/'model.zip').write_bytes(b'incomplete previous write')
        path = save_bundle(model, tmp_path, rank=0, world_size=1,
            run_id='test', wandb_id='same', config={})
        abandoned = list((tmp_path/'rank0').glob('.incomplete-*'))
        assert len(abandoned) == 1
        assert (abandoned[0]/'model.zip').read_bytes() == b'incomplete previous write'
        expected_random = (random.random(), np.random.random(), torch.rand(4))
        model._train_nosl_is_az(prepare_replay=False)
        recovered = AccumulatingMaskablePPO.load(path/'model.zip', device='cpu')
        recovered.configure_nosl_is_az_inner_only()
        restore_learner(recovered, path, rank=0, world_size=1)
        actual_random = (random.random(), np.random.random(), torch.rand(4))
        assert_nested_equal(expected_random, actual_random)
        for key in ('_nosl_is_az_fresh_samples', '_nosl_is_az_replay_prepare_stats'):
            setattr(recovered, key, copy.deepcopy(getattr(model, key)))
        recovered._train_nosl_is_az(prepare_replay=False)
        assert_nested_equal(model.policy.state_dict(), recovered.policy.state_dict())
        assert_nested_equal(model.policy.optimizer.state_dict(), recovered.policy.optimizer.state_dict())
        first, second = model._nosl_is_az_replay, recovered._nosl_is_az_replay
        assert (first.position, first.size, first.total_added) == (second.position, second.size, second.total_added)
        np.testing.assert_array_equal(first.sample_ids[:first.size], second.sample_ids[:second.size])
        with pytest.raises(ValueError, match='rank/world'):
            read_bundle(path, rank=1, world_size=2)
        with (path/'learner.pt').open('ab') as stream:
            stream.write(b'corruption')
        with pytest.raises(ValueError, match='checksum'):
            read_bundle(path, rank=0, world_size=1)
    finally:
        env.close()
