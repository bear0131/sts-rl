import numpy as np
import pytest
import torch

import env.core  # Select the configured native extension before importing it.
import slaythespire as sts
from rl.evaluation.strategies import DecisionRequest, OnlinePlayoutEvaluationStrategy
from rl.policies.token_regret import OnlineTokenRegretPolicy, improved_log_target
from rl.search.paired_token_rollout import PairedTokenBatch


pytestmark = pytest.mark.native


def test_regret_target_uses_measured_cost_and_conditions_on_tested_actions():
    row = dict(visits=[1, 5, 0], prior_logits=np.log([0.2, 0.6, 0.2]),
               values=[1., .99, 400.], return_units=100.)
    # One HP lost halves the prior odds for action 1, irrespective of its visits.
    np.testing.assert_allclose(np.exp(improved_log_target(row)), [.4, .6, 0.], atol=1e-6)
    row["values"][1] = 1.
    np.testing.assert_allclose(np.exp(improved_log_target(row)), [.25, .75, 0.], atol=1e-6)


def test_shared_search_root_and_guide_are_used_in_training_and_frozen_eval(combat_factory):
    env = combat_factory(["BASH", "STRIKE_RED", "DEFEND_RED"] * 3,
                         seed=190, combat_action_contract="autoregressive-v2")
    torch.manual_seed(91)
    net = OnlineTokenRegretPolicy(sts.small_playout_key_count()).eval()
    policy = None
    pool = sts.NeuralSearchBatch(1)
    try:
        batch = PairedTokenBatch([env], pool, net.collection_pair(collect_samples=True), net)
        snapshot = net.export_guide().copy()
        row, initial = batch.score_roots([env])[0]
        with torch.no_grad():
            net.source_score.weight.normal_(std=1.0)
        _, changed = batch.score_roots([env])[0]
        assert np.max(np.abs(changed - initial)) > 0.01
        np.testing.assert_array_equal(net.export_guide(), snapshot)
        req = DecisionRequest(env, {}, {"phase": "combat"}, 0, 9)
        choice = batch.decisions({0: ((), {"state_nonce": 9})})[0]["action"]
        sample, = batch.samples
        np.testing.assert_allclose(sample["prior_logits"], changed, atol=1e-6)
        visited = sample["visits"] > 0
        eligible = visited & (sample["values"] >= sample["values"][visited].max() - .005)
        best = changed[eligible].max()
        recommend = sample["recommended_action"]
        i = np.flatnonzero(sample["action_bits"] == recommend).item()
        assert eligible[i] and changed[i] == best
        policy = OnlinePlayoutEvaluationStrategy(net)
        policy.bind_environments([env])
        assert policy.decide([req])[0].action == choice
        assert policy._batch.samples == []
        assert policy.manifest()["lower_sha256"] == net.collection_pair(collect_samples=False)["sha256"]
    finally:
        if policy is not None:
            policy.unbind_environments()
        pool.clear()
