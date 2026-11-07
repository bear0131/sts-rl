"""Native counterfactual audit, no-search learning, and frozen evaluation."""
import numpy as np
import pytest
import torch
import slaythespire as sts

from env.search.small_playout import PairedWorldPlayoutStream
from rl.policies.search_budget import AdaptiveTokenRegretPolicy, AUDIT_PROBABILITY, BUDGETS
from rl.search.adaptive_budget import AdaptiveBudgetBatch
from rl.evaluation.strategies import DecisionRequest, OnlinePlayoutEvaluationStrategy

pytestmark = [pytest.mark.native, pytest.mark.timeout(90)]


def make(envs, *, training=True):
    torch.manual_seed(882)
    net = AdaptiveTokenRegretPolicy(sts.small_playout_key_count()).eval()
    pool = sts.NeuralSearchBatch(3)
    pair = net.collection_pair(collect_samples=training)
    pair['playout_depth'] = 12
    return AdaptiveBudgetBatch(envs, pool, pair, net), pool, net


def nonce_for(audit):
    return next(n for n in range(10000) if
                (np.random.default_rng(n ^ 0x4155444954).random() < AUDIT_PROBABILITY) == audit)


@pytest.mark.parametrize('audit', [False, True])
def test_zero_budget_has_no_fictitious_teacher_and_audits_do_not_change_execution(combat_factory, monkeypatch, audit):
    monkeypatch.setattr('rl.search.adaptive_budget.choose_budget', lambda *a, **kw: 0)
    env = combat_factory(['BASH', 'STRIKE_RED', 'DEFEND_RED'] * 3, seed=190,
                         combat_action_contract='autoregressive-v2')
    batch, pool, net = make([env])
    try:
        result = batch.decisions({0: ((), dict(state_nonce=nonce_for(audit)))})
        row, = batch.samples
        assert not row['budget_searched'] and row['budget_audit'] == audit
        assert not np.any(row['visits'])
        chosen = np.flatnonzero(row['action_bits'] == row['recommended_action']).item()
        assert row['prior_logits'][chosen] == max(row['prior_logits'])
        stats = batch.scalar_stats()
        assert stats['common/simulations'] == 0
        assert (stats['common/budget_total_simulations'] > sum(BUDGETS)) == audit
        data = net.prepare_samples(batch.samples, 1)
        inputs, targets, _ = data.batch(np.array([0]), 'cpu')
        terms = net.update_terms(inputs, targets)
        assert torch.isfinite(terms[0])
        assert terms[3].item() == 0  # No search-target KL, including audited zero decisions.
        terms[0].backward()
        if audit:
            assert net.budget_head[-1].bias.grad.abs().sum() > 0
        else:
            assert net.budget_head[-1].bias.grad.abs().sum() == 0
        assert not batch._contexts and result[0]['action'] >= 0
    finally:
        pool.clear()


def test_adaptive_stream_and_frozen_strategy_preserve_slots_and_no_audit(combat_factory, monkeypatch):
    monkeypatch.setattr('rl.search.adaptive_budget.choose_budget', lambda *a, **kw: 4)
    envs = [combat_factory(['BASH', 'STRIKE_RED', 'DEFEND_RED'] * 3, seed=s,
                          combat_action_contract='autoregressive-v2') for s in (190, 191)]
    batch, pool, net = make(envs, training=False)
    try:
        calls = {i: ((), dict(state_nonce=i + 44)) for i in range(2)}
        with PairedWorldPlayoutStream(batch, threads=3, batch_limit=1) as stream:
            stream.submit(calls)
            with pytest.raises(RuntimeError, match='outstanding'):
                stream.submit(calls)
            got = {}
            while stream.pending:
                got.update(stream.receive(timeout=.05))
        assert set(got) == {0, 1} and not batch.samples and not batch._contexts
        stats = batch.scalar_stats()
        assert stats['common/simulations'] == 4096
        assert stats['common/budget_audit_transitions'] == 0
        strategy = OnlinePlayoutEvaluationStrategy(net, threads=3)
        try:
            strategy.bind_environments(envs)
            decisions = strategy.decide([DecisionRequest(e, {}, {'phase': 'combat'}, i, 44)
                                         for i, e in enumerate(envs)])
            assert len(decisions) == 2
            assert strategy.manifest()['budget_config']['budgets'] == list(BUDGETS)
            assert strategy._batch.scalar_stats()['common/budget_audits'] == 0
        finally:
            strategy.unbind_environments()
    finally:
        pool.clear()


def test_dome_and_forced_roots_never_acquire_a_budget(combat_factory):
    dome = combat_factory(['BASH', 'STRIKE_RED', 'DEFEND_RED'] * 3, seed=190,
                          combat_action_contract='autoregressive-v2', relics=['Runic Dome'])
    forced = combat_factory(['WOUND'] * 10, seed=191, combat_action_contract='autoregressive-v2')
    batch, pool, net = make([dome, forced])
    try:
        out = batch.decisions({i: ((), dict(state_nonce=nonce_for(True))) for i in range(2)})
        assert len(out) == 2 and not batch.samples and not batch._contexts
        assert batch.stats['dome_policy_decisions'] == 1
        assert batch.stats['budget/forced_roots'] == 1
    finally:
        pool.clear()


@pytest.mark.parametrize('budget', [0, 37])
def test_fixed_reference_uses_same_frozen_model_and_reports_actual_cost(combat_factory, budget):
    env = combat_factory(['BASH', 'STRIKE_RED', 'DEFEND_RED'] * 3, seed=190,
                         combat_action_contract='autoregressive-v2')
    torch.manual_seed(882)
    net = AdaptiveTokenRegretPolicy(sts.small_playout_key_count()).eval()
    before = net.collection_pair(collect_samples=False)['sha256']
    strategy = OnlinePlayoutEvaluationStrategy(net, threads=3, fixed_budget=budget)
    try:
        strategy.bind_environments([env])
        strategy.decide([DecisionRequest(env, {}, {'phase': 'combat'}, 0, 44)])
        stats = strategy.manifest()['search_statistics']
        assert stats['common/budget_total_simulations'] == stats['common/simulations'] == budget
        assert stats['common/budget_audit_transitions'] == 0
        assert strategy.manifest()['fixed_budget_simulations'] == budget
        assert strategy.manifest()['lower_sha256'] == before
        assert not strategy._batch.samples
    finally:
        strategy.unbind_environments()


@pytest.mark.timeout(240)
@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_budget_production_payload_two_rounds_and_resume(tmp_path, monkeypatch, short_natural_game_env_type, device):
    from tests.integration.test_episode_rollout import _cohort_lifecycle
    # Tiny lifecycle cohorts deliberately audit every root to exercise nonzero
    # budget gradients; searches, targets, real learner and payload are unchanged.
    monkeypatch.setattr('rl.search.adaptive_budget.AUDIT_PROBABILITY', 1.)
    _cohort_lifecycle(tmp_path, monkeypatch, short_natural_game_env_type, device,
                      False, None, unified=True, independent=True, learned_budget=True)
