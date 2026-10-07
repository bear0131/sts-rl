"""Whole-run strategic PPO must not mutate the existing combat computation."""
import copy
import numpy as np
import pytest
import torch

from env.core.run_env import SlayTheSpireEnv
from rl.algorithms.accumulating_ppo import AccumulatingMaskablePPO
from rl.algorithms.upper_ppo import train_phase, update_upper
from rl.policies.entity_combat import EntityCombatPolicy
from rl.policies.entity_joint import EntityJointPolicy
from rl.rules.rule_vector_model import FROZEN_CARD_VECTOR_TABLE_SCHEMA
from env.core.cards_vocab import CARD_VOCAB
from rl.training.checkpoints import save_bundle, restore_learner
from rl.training.environments import TrainingStateWrapper
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.logger import configure

pytestmark = [pytest.mark.native, pytest.mark.training, pytest.mark.timeout(120)]


def make_joint(env, policy=EntityJointPolicy, card_vector_checkpoint=None):
    kwargs = dict(entity_config=dict(width=32, layers=1, heads=4, quantiles=8, auxiliary_loss_coef=0))
    if card_vector_checkpoint is not None:
        kwargs['entity_config']['card_vector_checkpoint'] = str(card_vector_checkpoint)
    if policy is EntityJointPolicy:
        kwargs['upper_config'] = dict(width=32, layers=1, heads=4, epochs=2, micro_batch_size=2)
    return AccumulatingMaskablePPO(policy, env, seed=7, n_steps=8, batch_size=4,
        n_epochs=1, device='cpu', policy_kwargs=kwargs, nosl_is_az=True, nosl_is_az_cpp=True,
        nosl_is_az_inner_only=True, nosl_is_az_upper_strategy='model' if policy is EntityJointPolicy else 'rules',
        nosl_is_az_simulations=4, nosl_is_az_depth=0, nosl_is_az_updates=1,
        nosl_is_az_batch_size=8, nosl_is_az_micro_batch_size=4, nosl_is_az_replay_capacity=128)


def test_frozen_card_vectors_are_shared_and_locked(tmp_path):
    vectors = torch.randn(len(CARD_VOCAB), 2, 12, generator=torch.Generator().manual_seed(19))
    vector_checkpoint = tmp_path / 'card-vectors.pt'
    torch.save(dict(schema=FROZEN_CARD_VECTOR_TABLE_SCHEMA,
                    card_vocab=list(CARD_VOCAB), vectors=vectors), vector_checkpoint)
    env = DummyVecEnv([lambda: TrainingStateWrapper(SlayTheSpireEnv(
        card_language=False, global_map_view=True,
        combat_action_contract='autoregressive-v2'))])
    model = make_joint(env, card_vector_checkpoint=vector_checkpoint)
    core = model.policy.core
    assert 'draw_card_ids' not in model.observation_space.spaces
    assert core.combat.card is core.upper_tokenizer.card
    assert tuple(core.combat.card.parameters()) == ()
    torch.testing.assert_close(core.combat.card.vectors, vectors, rtol=0, atol=0)
    assert core.combat.card_projection.in_features == 13
    assert core.upper_tokenizer.card_projection.in_features == 13
    before = core.combat.card.vectors.clone()
    for optimizer, parameters in (
            (model.policy.optimizer, core.combat.parameters()),
            (model.policy.upper_optimizer, core.upper_parameters())):
        optimizer.zero_grad()
        sum(parameter.square().sum() for parameter in parameters
            if parameter.requires_grad).backward()
        optimizer.step()
        torch.testing.assert_close(core.combat.card.vectors, before, rtol=0, atol=0)
    model.save(tmp_path / 'joint-frozen.zip')
    restored = AccumulatingMaskablePPO.load(tmp_path / 'joint-frozen.zip', device='cpu')
    torch.testing.assert_close(restored.policy.core.combat.card.vectors,
                               before, rtol=0, atol=0)
    assert restored.policy.core.combat.card is restored.policy.core.upper_tokenizer.card
    env.close()


def test_combat_unchanged_and_optimizers_independent(combat_factory, tmp_path):
    env = combat_factory(['BASH', 'STRIKE_RED', 'DEFEND_RED'], card_language=True,
                         combat_action_contract='autoregressive-v2')
    old = make_joint(env, EntityCombatPolicy)
    joint = make_joint(env)
    assert joint.reset_nosl_is_az_shared_embeddings(7) == 0
    joint.configure_nosl_is_az_inner_only()
    old_params = old.policy.core.combat.state_dict()
    for key, value in joint.policy.core.combat.state_dict().items():
        torch.testing.assert_close(value, old_params[key], rtol=0, atol=0)
    obs = {k: torch.as_tensor(v[None]) for k, v in env._obs().items()}
    old.policy.eval(); joint.policy.eval()
    with torch.no_grad():
        expected = old.policy.core.combat_actor_critic(obs)
        actual = joint.policy.core.actor_critic(obs)
    for x, y in zip(actual, expected):
        torch.testing.assert_close(x, y, rtol=0, atol=0)
    before = copy.deepcopy(joint.policy.core.combat.state_dict())
    upper_ids = {id(p) for p in joint.policy.core.upper_parameters()}
    assert not upper_ids & {id(p) for group in joint.policy.optimizer.param_groups for p in group['params']}
    with train_phase(joint, upper=True):
        loss = sum(p.square().sum() for p in joint.policy.core.upper_parameters())
        loss.backward(); joint.policy.upper_optimizer.step(); joint.policy.upper_optimizer.zero_grad()
    for k, value in joint.policy.core.combat.state_dict().items():
        torch.testing.assert_close(value, before[k], rtol=0, atol=0)
    joint.save(tmp_path / 'joint.zip')
    restored = AccumulatingMaskablePPO.load(tmp_path / 'joint.zip', device='cpu')
    assert restored.policy.upper_optimizer.state_dict()['state']
    assert not restored.policy.optimizer.state_dict()['state']
    from rl.policies.checkpoint import load_checkpoint
    shared_loaded = load_checkpoint(tmp_path / 'joint', device='cpu')
    assert shared_loaded.policy.upper_optimizer.state_dict()['state']
    for key, value in shared_loaded.policy.state_dict().items():
        torch.testing.assert_close(value, joint.policy.state_dict()[key], rtol=0, atol=0)
    # Distributed NOSL synchronizes after the decorated local gradient call.
    # Nested phase scopes must retain that gradient for the outer all-reduce.
    upper_before = copy.deepcopy(joint.policy.core.overworld.state_dict())
    with train_phase(joint):
        with train_phase(joint):
            loss = sum(p.square().sum() for p in joint.policy.core.combat.parameters()
                       if p.requires_grad)
            loss.backward()
        assert any(p.grad is not None for p in joint.policy.core.combat.parameters())
        assert all(p.grad is None for p in joint.policy.core.upper_parameters())
        joint.policy.optimizer.step()
    for key, value in joint.policy.core.overworld.state_dict().items():
        torch.testing.assert_close(value, upper_before[key], rtol=0, atol=0)


def test_actual_wholerun_upper_collection_update_and_bundle(tmp_path):
    env = DummyVecEnv([lambda: TrainingStateWrapper(SlayTheSpireEnv(
        card_language=True, global_map_view=True, combat_action_contract='autoregressive-v2'))])
    model = make_joint(env)
    model.configure_nosl_is_az_inner_only()
    model.set_logger(configure(str(tmp_path), []))
    # Two genuine collections with the standard whole-run start, not injected PPO rows.
    _, callback = model._setup_learn(16)
    callback.on_training_start({}, {})
    snapshots = []
    for _ in range(2):
        assert model.collect_rollouts(env, callback, model.rollout_buffer, n_rollout_steps=8)
        before = copy.deepcopy(model.policy.core.combat.state_dict())
        stats = update_upper(model)
        snapshots.append(stats)
        for k, value in model.policy.core.combat.state_dict().items():
            torch.testing.assert_close(value, before[k], rtol=0, atol=0)
    assert snapshots[0]['samples'] > 0 and snapshots[0]['updates'] > 0
    bundle = save_bundle(model, tmp_path/'bundle', rank=0, world_size=1,
        run_id='joint', wandb_id='joint', config={})
    old_pending = copy.deepcopy(model._upper_pending)
    loaded = AccumulatingMaskablePPO.load(bundle/'model.zip', env=env, device='cpu')
    restore_learner(loaded, bundle, rank=0, world_size=1)
    assert loaded._upper_episode_ids == model._upper_episode_ids
    assert [p is None for p in loaded._upper_pending] == [p is None for p in old_pending]
    for a, b in zip(loaded._upper_pending, old_pending):
        if a is not None:
            assert a['action'] == b['action'] and a['log_prob'] == b['log_prob']
            for key in a['observation']:
                np.testing.assert_array_equal(a['observation'][key], b['observation'][key])
    assert loaded.policy.upper_optimizer.state_dict()['state']
    env.close()
