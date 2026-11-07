import json

import numpy as np
import pytest
import torch

import env.core
import slaythespire as sts
from rl.algorithms.budget_calibration import BudgetSampleCollector, action_model_digest
from rl.evaluation.strategies import DecisionRequest, OnlinePlayoutEvaluationStrategy
from rl.policies.search_budget import AdaptiveTokenRegretPolicy, AUDIT_PROBABILITY
from rl.search.adaptive_budget import AdaptiveBudgetBatch

pytestmark = pytest.mark.native


def test_real_audit_collects_only_budget_features_and_preserves_action_parameters(combat_factory, tmp_path):
    env = combat_factory(["BASH", "STRIKE_RED", "DEFEND_RED"] * 3,
                         seed=190, combat_action_contract="autoregressive-v2")
    torch.manual_seed(91)
    net = AdaptiveTokenRegretPolicy(sts.small_playout_key_count()).eval()
    before = action_model_digest(net)
    path = tmp_path / "audit.npz"
    collector = BudgetSampleCollector(path, net)
    strategy = OnlinePlayoutEvaluationStrategy(net, threads=2, sample_collector=collector)
    nonce = next(n for n in range(4096)
                 if np.random.default_rng(n ^ 0x4155444954).random() < AUDIT_PROBABILITY)
    try:
        strategy.bind_environments([env])
        decision, = strategy.decide([DecisionRequest(env, {}, {"phase": "combat"}, 0, nonce)])
        assert env.action_masks()[decision.action]
        env.step(decision.action)
        assert strategy._batch.samples == []
    finally:
        strategy.unbind_environments()
    with np.load(path, allow_pickle=False) as data:
        assert data["features"].shape == (1, 100)
        assert data["targets"].shape == (1, 8)
        assert np.isfinite(data["features"]).all() and np.isfinite(data["targets"]).all()
        meta = json.loads(str(data["metadata"]))
        assert meta["audit_rows"] == meta["seen_roots"] == 1
        assert meta["episode_seeds"] == [0]
    assert action_model_digest(net) == before


def test_price_override_is_only_allowed_in_frozen_evaluation(combat_factory):
    env = combat_factory(["BASH", "STRIKE_RED", "DEFEND_RED"] * 3,
                         seed=191, combat_action_contract="autoregressive-v2")
    net = AdaptiveTokenRegretPolicy(sts.small_playout_key_count()).eval()
    pool = sts.NeuralSearchBatch(1)
    try:
        with pytest.raises(ValueError, match="frozen evaluation"):
            AdaptiveBudgetBatch([env], pool, net.collection_pair(collect_samples=True), net, cost_price=.1)
        batch = AdaptiveBudgetBatch([env], pool, net.collection_pair(collect_samples=False), net, cost_price=.1)
        assert batch.cost_price == .1
    finally:
        pool.clear()


def test_dataset_save_error_still_releases_native_pool(combat_factory, tmp_path, monkeypatch):
    env = combat_factory(["BASH", "STRIKE_RED", "DEFEND_RED"] * 3,
                         seed=192, combat_action_contract="autoregressive-v2")
    net = AdaptiveTokenRegretPolicy(sts.small_playout_key_count()).eval()
    collector = BudgetSampleCollector(tmp_path / "failed.npz", net)
    strategy = OnlinePlayoutEvaluationStrategy(net, threads=1, sample_collector=collector)
    strategy.bind_environments([env])

    def failed_save(lower):
        raise OSError("disk full")

    monkeypatch.setattr(collector, "save", failed_save)
    with pytest.raises(OSError, match="disk full"):
        strategy.unbind_environments()
    assert strategy._batch is strategy._pool is None
    assert strategy._indices == {}
