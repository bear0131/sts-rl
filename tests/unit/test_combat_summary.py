import numpy as np
import torch

from rl.policies.combat_summary import AdditiveCombatInputs, CombatSummary
from rl.policies.shared_wholerun import LearnedCardTable


def make_inputs():
    torch.manual_seed(20261111)
    return AdditiveCombatInputs(LearnedCardTable(12, 8), 9, width=8)


def test_sparse_moves_match_rebuild_and_keep_siblings_separate():
    net = make_inputs()
    ids = torch.tensor([1, 2, 3, 4])
    upgrades = torch.tensor([0., .2, .8, 0.])
    terms = net.card_terms(ids, upgrades)
    original = CombatSummary(torch.stack((terms[:2].sum(0), terms[2], terms[3]))[None],
                             torch.tensor([[[2.], [1.], [1.]]]), torch.zeros(1, 8))
    sibling = original.fork()
    child = original.fork()
    # Draw card 1 into hand, discard it; move upgraded card 3 to exhaust.
    child.move_cards(torch.tensor([0, 0, 0]), torch.tensor([0, -1, 1]),
                     torch.tensor([-1, 1, 2]), terms[[0, 0, 2]])
    expected = CombatSummary(torch.stack((terms[1], terms[0], terms[2:].sum(0)))[None],
                             torch.tensor([[[1.], [1.], [2.]]]), torch.zeros(1, 8))
    torch.testing.assert_close(child.piles, expected.piles)
    torch.testing.assert_close(net.features(child), net.features(expected))
    torch.testing.assert_close(sibling.piles, original.piles, rtol=0, atol=0)
    assert child.piles.data_ptr() != original.piles.data_ptr()


def test_counter_change_updates_only_its_identity_and_has_no_clamping():
    net = make_inputs()
    ids = torch.tensor([[1, 2, 0]])
    before = torch.tensor([[.2, .9, 0.]])
    summary = CombatSummary(torch.zeros(1, 3, 8), torch.zeros(1, 3, 1),
                             net.relic_terms(ids, before).sum(1))
    after = before.clone(); after[0, 1] = 31.7
    old = net.relic_terms(ids[:, 1], before[:, 1])
    new = net.relic_terms(ids[:, 1], after[:, 1])
    summary.replace_relics(torch.tensor([0]), old, new)
    torch.testing.assert_close(summary.relics, net.relic_terms(ids, after).sum(1))
    swapped = net.relic_terms(ids, after.flip(1)).sum(1)
    assert not torch.allclose(summary.relics, swapped)
    assert not net.relic_terms(torch.tensor([0]), torch.tensor([99.])).any()
    assert not torch.allclose(net.relic_terms(torch.tensor([1]), torch.tensor([100.])),
                              net.relic_terms(torch.tensor([1]), torch.tensor([101.])))


def test_repeated_upgrade_magnitude_and_delta_gradients():
    net = make_inputs()
    ids = torch.tensor([2, 2])
    base = net.card_terms(ids, torch.tensor([.2, .2]))
    upgraded = net.card_terms(ids, torch.tensor([.4, .8]))
    expected = net.card_terms(ids, torch.tensor([.4, .8])).sum(0)
    incremental = base.sum(0) + (upgraded - base).sum(0)
    params = tuple(net.parameters())
    a = torch.autograd.grad(incremental.square().sum(), params, allow_unused=True, retain_graph=True)
    b = torch.autograd.grad(expected.square().sum(), params, allow_unused=True)
    torch.testing.assert_close(incremental, expected)
    assert not torch.allclose(upgraded[0], upgraded[1])
    for left, right in zip(a, b, strict=True):
        assert (left is None) == (right is None)
        if left is not None:
            torch.testing.assert_close(left, right, atol=2e-6, rtol=2e-5)


def test_frozen_relic_identity_preserves_active_flags_and_unbounded_counters():
    net = make_inputs()
    ids = torch.tensor([[1, 0, 2, 3]])
    static = (net.relic_identity(ids) * ids.ne(0)[..., None]).sum(1)
    original = static.clone()
    values = []
    for counters in (torch.tensor([[1., 99., .2, 0.]]),
                     torch.tensor([[0., 99., .2, 0.]]),
                     torch.tensor([[0., 99., 100.1, -.4]])):
        combined = static + (net.relic_counter_terms(ids, counters) * ids.ne(0)[..., None]).sum(1)
        torch.testing.assert_close(combined, net.relic_terms(ids, counters).sum(1), rtol=3e-5, atol=3e-6)
        values.append(combined)
    assert not torch.equal(values[0], values[1])
    assert not torch.equal(values[1], values[2])
    torch.testing.assert_close(static, original, rtol=0, atol=0)


def test_long_update_stream_does_not_accumulate_large_fp32_error():
    net = make_inputs()
    with torch.no_grad():
        terms = net.card_terms(torch.tensor([1, 2]), torch.tensor([0., .6]))
        state = CombatSummary(torch.stack((terms.sum(0), torch.zeros(8), torch.zeros(8)))[None],
                              torch.tensor([[[2.], [0.], [0.]]]), torch.zeros(1, 8))
        expected = state.fork()
        for _ in range(2000):
            state.move_cards(torch.tensor([0, 0]), torch.tensor([0, 0]), torch.tensor([1, 2]), terms)
            state.move_cards(torch.tensor([0, 0]), torch.tensor([1, 2]), torch.tensor([0, 0]), terms)
        torch.testing.assert_close(state.piles, expected.piles, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(state.pile_counts, expected.pile_counts, rtol=0, atol=0)


def test_frozen_cpu_tables_sparse_events_match_differentiable_reference():
    net = make_inputs()
    tables = net.export_inference_tables()
    obs = {p + '_counts': np.zeros(12, np.float32) for p in ('draw', 'discard', 'exhaust')}
    obs.update(pile_upgraded_counts=np.zeros((3, 12), np.float32),
        pile_upgrade_sums=np.zeros((3, 12), np.float32), relics=np.array([1, 2, 0]),
        relic_counters=np.array([.2, .9, 0.], np.float32))
    obs['draw_counts'][2] = 2
    obs['pile_upgraded_counts'][0, 2] = 2
    obs['pile_upgrade_sums'][0, 2] = .8
    state = tables.new_state(obs)
    sibling = state.fork()
    original = sibling.piles.copy()
    rng = np.random.default_rng(20261111)
    for step in range(2000):
        source, dest = (0, 1) if step % 2 == 0 else (1, 0)
        zones, ids = np.array([source, dest]), np.array([2, 2])
        counts, upgrades = np.array([-2., 2.], np.float32), np.array([-.8, .8], np.float32)
        state.card_events(zones, ids, counts, counts, upgrades)
        old = obs['relic_counters'].copy()
        obs['relic_counters'][1] = rng.integers(0, 10000) / 10
        state.relic_events(obs['relics'][1:2], old[1:2], obs['relics'][1:2], obs['relic_counters'][1:2])
    tensor = {k: torch.from_numpy(v[None]) for k, v in obs.items()}
    with torch.no_grad():
        expected = net.rebuild(tensor)
    np.testing.assert_allclose(state.piles, expected.piles[0], rtol=1e-5, atol=1e-5)
    # Repeated cancellation of large counters has FP32 roundoff; bound the
    # absolute error rather than pretending algebraic equivalence is bitwise.
    np.testing.assert_allclose(state.relics, expected.relics[0], rtol=2e-4, atol=.005)
    np.testing.assert_array_equal(sibling.piles, original)
    assert tables.card_base is sibling.tables.card_base
    assert not tables.card_base.flags.writeable
    events = sibling.events
    sibling.update({**obs, 'relic_counters': np.array([.2, .9, 0.], np.float32)})
    assert sibling.events == events  # unchanged observation performs no events
