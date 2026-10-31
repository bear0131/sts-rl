from types import SimpleNamespace

import numpy as np
import pytest
import torch

from env.core import action_space as A, vocab as V
from rl.training import reward_monitor as M


def scene(gold=(.1, .2), potions=(3, 3), *, potion_legal=True):
    obs = dict(phase=np.array([[M.REWARD_PHASE]]),
               reward_gold=np.array([list(gold) + [0] * (2-len(gold))]),
               reward_potion_ids=np.array([list(potions) + [0] * (5-len(potions))]))
    mask = np.zeros((1, A.N_ACTIONS), dtype=bool)
    mask[0, A.seg_base('PROCEED')] = True
    mask[0, A.seg_base('REWARD_GOLD'):A.seg_base('REWARD_GOLD')+len(gold)] = True
    if potion_legal:
        mask[0, A.seg_base('REWARD_POTION'):A.seg_base('REWARD_POTION')+len(potions)] = True
    return obs, mask


def step(model, state, action, info=None):
    obs, mask = state
    M.record_step(model, obs, np.array([action]), mask,
                  (np.array([0]), np.array([[.3, .2]])), [info or {}])


def test_repeated_reward_decisions_count_each_resolved_item_once():
    model = SimpleNamespace()
    M.begin_collection(model)
    step(model, scene(), A.seg_base('REWARD_CARD'))
    step(model, scene(), A.seg_base('REWARD_GOLD'))
    step(model, scene(gold=(.2,)), A.seg_base('REWARD_SKIP'))
    step(model, scene(gold=(.2,)), A.seg_base('PROCEED'))
    out = M.metrics(M.counts(model))
    assert out['reward_gold_resolved'] == 2
    assert out['reward_gold_taken'] == 1
    assert out['reward_gold_pickup_rate'] == .5
    assert out['reward_potion_resolved'] == 2  # Identical bottles remain distinct.
    assert out['reward_potion_pickup_rate'] == 0
    assert out['reward_gold_policy_decisions'] == 4
    assert out['reward_gold_policy_probability'] == pytest.approx(.3)


def test_unclaimable_potions_are_not_zero_probability_policy_decisions():
    model = SimpleNamespace()
    step(model, scene(potion_legal=False), A.seg_base('PROCEED'))
    out = M.metrics(M.counts(model))
    assert out['reward_potion_unavailable_left'] == 2
    assert out['reward_potion_pickup_rate'] == 0
    assert out['reward_potion_policy_decisions'] == 0
    assert 'reward_potion_policy_probability' not in out


def test_remaining_rewards_cross_collection_or_restore_without_double_count():
    model = SimpleNamespace()
    step(model, scene(gold=(), potions=(3, 3)), A.seg_base('REWARD_POTION'))
    first = M.counts(model)
    # The checkpoint needs only the real environment's remaining one bottle.
    resumed = SimpleNamespace()
    M.begin_collection(resumed)
    step(resumed, scene(gold=(), potions=(3,)), A.seg_base('PROCEED'))
    combined = {key: value + M.counts(resumed)[key] for key, value in first.items()}
    assert M.metrics(combined)['reward_potion_pickup_rate'] == .5
    assert combined['reward_potion_resolved'] == 2
    M.begin_collection(resumed)
    assert all(v == 0 for v in M.counts(resumed).values())


@pytest.mark.parametrize('info', [{'simulator_transition': False},
                                  {'TimeLimit.truncated': True}, {'dead_screen': True}])
def test_prefix_or_censored_steps_do_not_fabricate_uptake(info):
    model = SimpleNamespace()
    step(model, scene(), A.seg_base('PROCEED'), info)
    assert all(v == 0 for v in M.counts(model).values())


def test_rank_ratios_use_global_counts_and_missing_data_stays_missing():
    raw = M.counts(SimpleNamespace())
    # Rank 0 takes 1/1, rank 1 takes 0/9: their reduced means are .5/5.
    raw.update(reward_gold_taken=.5, reward_gold_resolved=5.,
               reward_gold_policy_probability_sum=.9, reward_gold_policy_decisions=5.)
    out = M.metrics(raw, world_size=2)
    assert out['reward_gold_pickup_rate'] == .1
    assert out['reward_gold_resolved'] == 10
    assert out['reward_gold_policy_probability'] == pytest.approx(.18)
    assert 'reward_potion_pickup_rate' not in out
    assert 'reward_potion_policy_probability' not in out


def test_existing_masked_distribution_is_read_without_forward_or_rng():
    probs = torch.zeros((2, A.N_ACTIONS))
    probs[0, A.seg_base('REWARD_GOLD'):A.seg_base('REWARD_GOLD')+2] = torch.tensor([.1, .2])
    probs[0, A.seg_base('REWARD_POTION')] = .4
    probs[0, A.seg_base('PROCEED')] = .3
    probs[1, A.seg_base('PROCEED')] = 1
    distribution = SimpleNamespace(distribution=SimpleNamespace(probs=probs))
    rng = torch.get_rng_state().clone()
    indices, values = M.policy_probabilities(distribution, {
        'phase': np.array([[M.REWARD_PHASE], [V.PHASE_VOCAB_INDEX['map']]])})
    np.testing.assert_array_equal(indices, [0])
    np.testing.assert_allclose(values, [[.3, .4]])
    assert torch.equal(torch.get_rng_state(), rng)
