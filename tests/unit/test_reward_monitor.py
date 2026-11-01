from types import SimpleNamespace

import numpy as np
import pytest
import torch

from env.core import action_space as A, vocab as V
from rl.training import reward_monitor as M


def scene(gold=(.1, .2), potions=(3, 3), *, potion_legal=True, relics=(), sapphire=False):
    obs = dict(phase=np.array([[M.REWARD_PHASE]]),
               reward_relic_ids=np.array([list(relics)+[0]*(3-len(relics))]),
               reward_keys=np.array([[0,int(sapphire)]]),
               reward_gold=np.array([list(gold) + [0] * (2-len(gold))]),
               reward_potion_ids=np.array([list(potions) + [0] * (5-len(potions))]))
    mask = np.zeros((1, A.N_ACTIONS), dtype=bool)
    mask[0, A.seg_base('PROCEED')] = True
    mask[0, A.seg_base('REWARD_GOLD'):A.seg_base('REWARD_GOLD')+len(gold)] = True
    mask[0,A.seg_base('REWARD_RELIC'):A.seg_base('REWARD_RELIC')+len(relics)] = True
    mask[0,A.seg_base('REWARD_KEY')] = sapphire
    if potion_legal:
        mask[0, A.seg_base('REWARD_POTION'):A.seg_base('REWARD_POTION')+len(potions)] = True
    return obs, mask


def step(model, state, action, info=None):
    obs, mask = state
    M.record_step(model, obs, np.array([action]), mask,
                  (np.array([0]), np.array([[.3, .2, .4]])), [info or {}])


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
    probs[0, A.seg_base('REWARD_RELIC')] = .2
    probs[0, A.seg_base('PROCEED')] = .1
    probs[1, A.seg_base('PROCEED')] = 1
    distribution = SimpleNamespace(distribution=SimpleNamespace(probs=probs))
    rng = torch.get_rng_state().clone()
    indices, values = M.policy_probabilities(distribution, {
        'phase': np.array([[M.REWARD_PHASE], [V.PHASE_VOCAB_INDEX['map']]])})
    np.testing.assert_array_equal(indices, [0])
    np.testing.assert_allclose(values, [[.3, .4, .2]])
    assert torch.equal(torch.get_rng_state(), rng)


def test_relic_claim_key_exchange_and_leave_resolve_each_offer_once():
    model = SimpleNamespace()
    step(model, scene(relics=(1, 2), sapphire=True), A.seg_base('REWARD_GOLD'))
    assert M.counts(model)['reward_relic_resolved'] == 0
    step(model, scene(relics=(1, 2), sapphire=True), A.seg_base('REWARD_KEY'))
    step(model, scene(relics=(1,)), A.seg_base('REWARD_RELIC'))
    step(model, scene(), A.seg_base('PROCEED'))
    out = M.metrics(M.counts(model))
    assert out['reward_relic_taken'] == 1 and out['reward_relic_resolved'] == 2
    assert out['reward_relic_key_exchanged'] == 1 and out['reward_relic_pickup_rate'] == .5
    assert out['reward_relic_policy_probability'] == pytest.approx(.4)
    assert out['reward_relic_policy_decisions'] == 3
    M.begin_collection(model)
    step(model, scene(relics=(1, 1)), A.seg_base('PROCEED'))
    assert M.metrics(M.counts(model))['reward_relic_resolved'] == 2


@pytest.mark.parametrize('phase', ['shop', 'boss_relic', 'chest', 'event'])
def test_other_relic_acquisition_phases_are_not_free_reward_claims(phase):
    distribution = SimpleNamespace(distribution=SimpleNamespace(probs=torch.zeros((1,A.N_ACTIONS))))
    indices,masses=M.policy_probabilities(distribution,{'phase':np.array([[V.PHASE_VOCAB_INDEX[phase]]])})
    assert len(indices)==0 and masses.shape==(0,3)
