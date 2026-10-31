from types import SimpleNamespace

import numpy as np
import pytest
import torch

from env.core import action_space as A, vocab as V
from rl.training import campfire_monitor as M


def record(model, choice, *, legal=('rest', 'smith', 'recall'), info=None):
    mask = np.zeros((1, A.N_ACTIONS), dtype=bool)
    mask[0, [M.OPTIONS[name] for name in legal]] = True
    probabilities = np.zeros((1, len(M.OPTIONS)))
    for column, name in enumerate(M.OPTIONS):
        probabilities[0, column] = 1 / len(legal) if name in legal else 0
    M.record_step(model, [M.OPTIONS.get(choice, choice)], mask,
                  (np.array([0]), probabilities), [info or {}])


def test_unlocked_options_have_their_own_denominator_and_missing_is_not_zero():
    model = SimpleNamespace()
    record(model, 'rest')
    record(model, 'lift', legal=('rest', 'lift'))
    out = M.metrics(M.counts(model))
    assert out['campfire_decisions'] == 2
    assert out['campfire_rest_choice_rate'] == .5
    assert out['campfire_lift_choice_rate'] == 1
    assert out['campfire_lift_share'] == .5
    assert out['campfire_rest_policy_probability'] == pytest.approx((1/3+.5)/2)
    assert out['campfire_smith_choice_rate'] == 0  # Legal, but not chosen.
    assert 'campfire_dig_choice_rate' not in out  # Never legal.
    assert 'campfire_dig_policy_probability' not in out


@pytest.mark.parametrize('name', M.OPTIONS)
def test_every_named_option_and_skip_counts_once(name):
    model = SimpleNamespace()
    record(model, name, legal=(name,))
    out = M.metrics(M.counts(model))
    assert out[f'campfire_{name}_selected'] == out['campfire_decisions'] == 1


@pytest.mark.parametrize('info', [{'simulator_transition': False},
                                  {'TimeLimit.truncated': True}, {'dead_screen': True}])
def test_uncommitted_or_censored_steps_do_not_count(info):
    model = SimpleNamespace()
    record(model, 'rest', info=info)
    assert not any(M.counts(model).values())


def test_potion_action_and_illegal_choice_do_not_resolve_a_campfire():
    model = SimpleNamespace()
    record(model, A.seg_base('USE_POTION'))
    record(model, 'dig')
    assert not any(M.counts(model).values())


def test_global_unequal_rank_counts_and_collection_resume_reset():
    a, b = SimpleNamespace(), SimpleNamespace()
    record(a, 'rest', legal=('rest', 'smith'))
    for _ in range(9):
        record(b, 'smith', legal=('rest', 'smith'))
    means = {k: (v+M.counts(b)[k])/2 for k, v in M.counts(a).items()}
    out = M.metrics(means, world_size=2)
    assert out['campfire_decisions'] == 10
    assert out['campfire_rest_choice_rate'] == .1
    assert out['campfire_rest_policy_probability'] == .5
    # Restoring a model's last metrics must not carry them into a new collection.
    restored = SimpleNamespace(_upper_campfire_counts=M.counts(a))
    M.begin_collection(restored)
    assert not any(M.counts(restored).values())
    assert 'campfire_rest_choice_rate' not in M.metrics(M.counts(restored))


def test_existing_distribution_excludes_non_campfire_mass_without_rng_or_mutation():
    probs = torch.zeros((2, A.N_ACTIONS))
    probs[0, M.OPTIONS['rest']] = .2
    probs[0, M.OPTIONS['smith']] = .3
    probs[0, A.seg_base('USE_POTION')] = .5
    probs[1, M.OPTIONS['skip']] = 1
    before = probs.clone()
    distribution = SimpleNamespace(distribution=SimpleNamespace(probs=probs))
    rng = torch.get_rng_state().clone()
    indices, values = M.policy_probabilities(distribution, {
        'phase': np.array([[M.REST_PHASE], [V.PHASE_VOCAB_INDEX['card_select']]])})
    np.testing.assert_array_equal(indices, [0])
    np.testing.assert_allclose(values, [[.4, .6, 0, 0, 0, 0, 0]])
    assert torch.equal(probs, before) and torch.equal(torch.get_rng_state(), rng)
    empty = M.policy_probabilities(distribution, {'phase': np.array([[0], [0]])})
    assert empty[1].shape == (0, 7)
