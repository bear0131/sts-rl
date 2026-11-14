import torch
import numpy as np
import pytest
from types import SimpleNamespace
from dataclasses import asdict

from rl.algorithms.search_auxiliary import improved_prior, search_kl


@pytest.fixture
def search_row_buffer():
    from gymnasium import spaces
    from rl.algorithms.search_auxiliary import SearchRolloutBuffer
    return SearchRolloutBuffer(3, spaces.Dict({
        'identity': spaces.Box(-1000, 1000, (2,), dtype=np.float32),
        'phase': spaces.Discrete(8),
    }), spaces.Discrete(4), n_envs=3, device='cpu', gamma=1., gae_lambda=1.)


def _real_rows(times, indices):
    times, indices = np.asarray(times), np.asarray(indices)
    action = (times + indices) % 3
    masks = np.ones((len(indices), 4), dtype=bool)
    masks[:, 3] = False
    searched = (times + indices) % 2 == 0
    logp = np.where(searched, 0., -.3).astype(np.float32)
    target = np.array([np.log(.2), np.log(.3), np.log(.5), -np.inf], np.float32)
    teachers = {int(e): target.copy() for e, used in zip(indices, searched) if used}
    executed = {int(e): (int(a), float(p)) for e, a, p, used in zip(indices, action, logp, searched) if used}
    data = (dict(identity=np.stack((times, indices), -1).astype(np.float32), phase=(times+indices)%8),
            action, (1.+times/3+indices/7).astype(np.float32),
            (times == 0) | ((indices == 1) & (times == 2)),
            torch.as_tensor((times/5+indices/9).astype(np.float32)), torch.as_tensor(logp))
    return data, dict(action_masks=masks, search_targets=teachers, search_actions=executed)


def test_full480_ready_rows_preserve_ppo_targets_shuffle_and_owned_storage(search_row_buffer):
    from copy import deepcopy
    rows = search_row_buffer
    reference = deepcopy(rows)
    retained = None
    for iteration in range(2):
        rows.reset(); reference.reset()
        for time in range(3):
            data, meta = _real_rows([time]*3, [0, 1, 2])
            for env, teacher in meta['search_targets'].items():
                reference.add_search_target(env, teacher)
            for env, behavior in meta['search_actions'].items():
                reference.search_actions[(time, env)] = behavior
            reference.add(*data, action_masks=meta['action_masks'])
        for indices in ([2], [0, 1], [2], [0], [1, 2], [1], [0]):
            data, meta = _real_rows(rows.row_positions[indices], indices)
            rows.add_rows(indices, *data, **meta)
            # Every published observation/teacher/result owns its memory.
            for value in data[0].values():
                value.fill(99)
            for value in meta['search_targets'].values():
                value.fill(99)
            data[4].fill_(99); data[5].fill_(99)
            assert rows.pos == rows.row_positions.min()
            assert rows.full == bool((rows.row_positions == 3).all())
        for buffer in (rows, reference):
            buffer.compute_returns_and_advantage(torch.tensor([.2, .3, .4]), np.array([False, True, False]))
        for key in reference.observations:
            np.testing.assert_array_equal(rows.observations[key], reference.observations[key])
        for key in ('actions', 'rewards', 'episode_starts', 'values', 'log_probs',
                    'advantages', 'returns', 'action_masks'):
            np.testing.assert_array_equal(getattr(rows, key), getattr(reference, key))
        assert rows.search_actions == reference.search_actions
        for key, target in reference.search_targets.items():
            np.testing.assert_array_equal(rows.search_targets[key], target)
        np.random.seed(20261111)
        actual = list(rows.get(4))
        np.random.seed(20261111)
        expected = list(reference.get(4))
        for a, b in zip(actual, expected, strict=True):
            for key in a.observations:
                torch.testing.assert_close(a.observations[key], b.observations[key], rtol=0, atol=0)
            for key in a._fields[1:]:
                torch.testing.assert_close(getattr(a, key), getattr(b, key), rtol=0, atol=0)
        if retained is not None:
            torch.testing.assert_close(retained[0], retained[1], rtol=0, atol=0)
        retained = actual[0].observations['identity'], actual[0].observations['identity'].clone()


@pytest.mark.parametrize('invalid', ['duplicate', 'missing_observation', 'illegal_action', 'behavior', 'nan', 'teacher'])
def test_full480_invalid_ready_packet_does_not_partially_write(search_row_buffer, invalid):
    rows = search_row_buffer
    indices = [0, 1]
    data, meta = _real_rows([0, 0], indices)
    if invalid == 'duplicate':
        indices = [0, 0]
    elif invalid == 'missing_observation':
        del data[0]['identity']
    elif invalid == 'illegal_action':
        data[1][0] = 3
    elif invalid == 'behavior':
        meta['search_actions'][0] = (1, 0.)
    elif invalid == 'nan':
        data[4][0] = torch.nan
    else:
        meta['search_targets'][0][0] = np.nan
    with pytest.raises(ValueError):
        rows.add_rows(indices, *data, **meta)
    assert rows.pos == 0 and not rows.full and not rows.row_positions.any()
    assert not rows.search_targets and not rows.search_actions
    assert not rows.observations['identity'].any() and not rows.values.any()
    assert rows.action_masks.all()


def test_full480_rows_cannot_overrun_or_mix_storage_protocols(search_row_buffer):
    rows = search_row_buffer
    for time in range(3):
        data, meta = _real_rows([time], [2])
        rows.add_rows([2], *data, **meta)
    assert not rows.full and rows.pos == 0
    with pytest.raises(ValueError, match='completed'):
        rows.add_rows([2], *data, **meta)
    all_data, all_meta = _real_rows([0]*3, [0, 1, 2])
    with pytest.raises(ValueError, match='mix'):
        rows.add(*all_data, action_masks=all_meta['action_masks'])
    with pytest.raises(ValueError, match='prefill'):
        rows.prefill_observation(all_data[0])
    with pytest.raises(AssertionError, match='complete'):
        next(rows.get())
    rows.reset()
    rows.add(*all_data, action_masks=all_meta['action_masks'])
    with pytest.raises(ValueError, match='mix'):
        rows.add_rows([2], *data, **meta)


def test_equal_values_preserve_prior_and_illegal_atoms_have_no_gradient():
    old = torch.tensor([[1., 4., -3.]], requires_grad=True)
    legal = torch.tensor([[True, False, True]])
    q = improved_prior(old, torch.tensor([[.2, float('nan'), .2]]), legal, torch.tensor([2]), .1)
    expected = old.detach().masked_fill(~legal, -torch.inf).log_softmax(-1)
    torch.testing.assert_close(q, expected)
    current = torch.tensor([[.1, 999., -.1]], requires_grad=True)
    loss = search_kl(current, q, legal).sum()
    loss.backward()
    assert torch.isfinite(current.grad).all()
    assert current.grad[0, 1] == 0
    assert old.grad is None and not q.requires_grad


def test_more_worlds_strengthen_bounded_improvement_and_constant_shift_cancels():
    logits = torch.tensor([[.5, 0.]])
    legal = torch.ones(1, 2, dtype=torch.bool)
    values = torch.tensor([[0., 1.]])
    low = improved_prior(logits, values, legal, torch.tensor([1]), .2)
    high = improved_prior(logits, values, legal, torch.tensor([20]), .2)
    shifted = improved_prior(logits, values + 100, legal, torch.tensor([20]), .2)
    assert high[0, 1] > low[0, 1]
    torch.testing.assert_close(high, shifted)
    odds_change = (high[0, 1] - high[0, 0]) - (logits[0, 1] - logits[0, 0])
    assert 0 < odds_change <= 4 * 20 / 24 + 1e-6  # FP32 log-softmax rounding


def test_auxiliary_normalizes_global_searched_rows_before_physical_chunks():
    from rl.algorithms.search_auxiliary import SearchTrainingConfig, prepare_search_batch, auxiliary_loss
    logits = torch.tensor([[.5, -.2], [.1, .4], [.2, .9], [.8, -.2]], requires_grad=True)
    target = torch.tensor([[.1, .9], [.8, .2], [.5, .5], [.4, .6]]).log()
    valid = torch.tensor([0., 1., 0., 0.])
    obs = dict(_search_target=target, _search_valid=valid)
    model = SimpleNamespace(deck_search_config=asdict(SearchTrainingConfig(coefficient=.1)))
    batch = SimpleNamespace(observations=obs, advantages=torch.ones(4))
    # Across four ranks only this rank has one searched row: the mean count
    # is .25, and must not be rounded/clamped up to one.
    prepare_search_batch(model, batch, lambda x: dict(rows=4., search_count=.25))
    whole = auxiliary_loss(model, obs, logits, torch.ones_like(logits, dtype=torch.bool))
    chunks = sum(auxiliary_loss(model, {k: v[i:i+1] for k, v in obs.items()}, logits[i:i+1],
        torch.ones_like(logits[i:i+1], dtype=torch.bool)) / 4 for i in range(4))
    expected = .1 * search_kl(logits[1:2], target[1:2], torch.ones(1, 2, dtype=torch.bool)).sum()
    torch.testing.assert_close(whole / 4, expected)  # rank gradient average
    torch.testing.assert_close(chunks, whole)
    torch.testing.assert_close(torch.autograd.grad(chunks, logits, retain_graph=True)[0],
                               torch.autograd.grad(whole, logits)[0])
