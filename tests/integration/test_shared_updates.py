"""A real replay batch follows the same learner/Adam path through local and ring adapters."""
import copy
import numpy as np
import pytest
import torch

from rl.algorithms.accumulating_ppo import AccumulatingMaskablePPO
from rl.training.distributed_sb3 import distributed_nosl_is_az_update
from tests.integration.test_training_smoke import train_once, assert_nested_equal

pytestmark = [pytest.mark.native, pytest.mark.training]


@pytest.mark.parametrize('entity', [False, True, 'autoregressive'])
def test_single_rank_ring_matches_local_replay_optimizer(tmp_path, short_combat_env_type, entity):
    if entity:
        from gymnasium.wrappers import TimeLimit
        from rl.policies.entity_combat import EntityCombatPolicy
        env = TimeLimit(short_combat_env_type(combat_action_contract='autoregressive-v1' if entity == 'autoregressive' else 'joint-v2',card_language=True),12)
        model = AccumulatingMaskablePPO(EntityCombatPolicy,env,device='cpu',seed=42,
            n_steps=16,batch_size=8,nosl_is_az=True,nosl_is_az_cpp=True,nosl_is_az_inner_only=True,
            nosl_is_az_upper_strategy='rules',nosl_is_az_simulations=4,nosl_is_az_depth=0,
            nosl_is_az_replay_capacity=128,nosl_is_az_batch_size=16,nosl_is_az_micro_batch_size=8,
            policy_kwargs=dict(entity_config=dict(width=32,layers=1,heads=4,quantiles=8)))
        model.learn(32)
    else:
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

        if entity:
            # Real Adam momentum exists before the extra value update. Ring
            # unpack fills unused gradients with zeros, so None must be restored.
            heads = list(ring_model.policy.core.policy_only_parameters())
            before = [p.detach().clone() for p in heads]
            adam = [copy.deepcopy(ring_model.policy.optimizer.state.get(p, {})) for p in heads]
            assert any(s.get('step', 0) > 0 for s in adam)
            assert any('exp_avg' in s and s['exp_avg'].abs().sum() > 0 for s in adam)
            value = ring_model.policy.core.combat.distribution.win.weight
            value_before = value.detach().clone()
            rng = np.random.get_state(), torch.get_rng_state()
            model._train_nosl_is_az(prepare_replay=False,update_count=1,value_only_updates=1)
            np.random.set_state(rng[0]); torch.set_rng_state(rng[1])
            extra = distributed_nosl_is_az_update(ring_model, rank=0, world_size=1, peers=[],
                key=tmp_path/'unused-key',control_dir=tmp_path,local_root=tmp_path,
                remote_root=str(tmp_path),run_id='loopback',update=0,
                ring_transport=LoopbackRing(),barrier_timeout=2,prepare_replay=False,
                update_count=1,value_only_updates=1)
            assert extra['value_only_updates'] == 1 and extra['joint_updates'] == 0
            for p, old, state in zip(heads, before, adam):
                assert torch.equal(p, old)
                assert_nested_equal(ring_model.policy.optimizer.state.get(p, {}), state)
            assert not torch.equal(value, value_before)
            assert_nested_equal(model.policy.state_dict(), ring_model.policy.state_dict())
            assert_nested_equal(model.policy.optimizer.state_dict(), ring_model.policy.optimizer.state_dict())

        numpy_rng, torch_rng = np.random.get_state(), torch.get_rng_state()
        budget = dict(update_count=3,value_only_updates=2) if entity else {}
        local = model._train_nosl_is_az(prepare_replay=False, **budget)
        np.random.set_state(numpy_rng)
        torch.set_rng_state(torch_rng)
        ring = distributed_nosl_is_az_update(ring_model, rank=0, world_size=1, peers=[],
            key=tmp_path/'unused-key', control_dir=tmp_path, local_root=tmp_path,
            remote_root=str(tmp_path), run_id='loopback', update=0,
            ring_transport=LoopbackRing(), barrier_timeout=2, prepare_replay=False, **budget)
        assert local['updates'] == ring['updates'] == (3 if entity else 2)
        assert_nested_equal(model.policy.state_dict(), ring_model.policy.state_dict())
        if entity:
            # During JOINT updates, unused sparse subset branches receive a
            # zero collective gradient, whereas local autograd leaves None.
            # Shared/active states must match; newly installed states may only
            # belong to those unused heads and have exactly zero momentum.
            for (name, p), (other_name, q) in zip(model.policy.named_parameters(), ring_model.policy.named_parameters()):
                assert name == other_name
                left = model.policy.optimizer.state.get(p, {})
                right = ring_model.policy.optimizer.state.get(q, {})
                if not left and right:
                    assert 'complete_head.subset.' in name
                    assert right['step'] == 1
                    assert torch.count_nonzero(right['exp_avg']) == 0
                    assert torch.count_nonzero(right['exp_avg_sq']) == 0
                else:
                    assert_nested_equal(left, right)
        else:
            assert_nested_equal(model.policy.optimizer.state_dict(), ring_model.policy.optimizer.state_dict())
    finally:
        env.close()
