import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from rl.algorithms.budget_calibration import (
    SIDECAR_CONTRACT, BudgetFitConfig, action_model_digest, attach_budget_head,
    fit_budget_head, frozen_policy_digest, save_budget_head,
)
from rl.policies.search_budget import budget_manifest, choose_budget
from rl.policies.token_regret import OnlineTokenRegretPolicy


def model():
    torch.manual_seed(227)
    policy = torch.nn.Module()
    policy.core = torch.nn.Linear(4, 4)
    policy.small_playout = OnlineTokenRegretPolicy(785).eval()
    return SimpleNamespace(policy=policy)


def test_head_sidecar_preserves_original_parameters_and_rejects_wrong_base(tmp_path):
    original = model()
    before = frozen_policy_digest(original.policy)
    guide = original.policy.small_playout.export_guide().copy()
    lower = attach_budget_head(original, base_sha256="a" * 64)
    np.testing.assert_array_equal(lower.export_guide(), guide)
    assert frozen_policy_digest(original.policy) == before
    path = tmp_path / "head.pt"
    save_budget_head(original, path)
    second = model()
    attach_budget_head(second, base_sha256="a" * 64, sidecar=path)
    assert frozen_policy_digest(second.policy) == before
    for name, value in lower.state_dict().items():
        torch.testing.assert_close(value, second.policy.small_playout.state_dict()[name], rtol=0, atol=0)
    with pytest.raises(ValueError, match="incompatible"):
        attach_budget_head(model(), base_sha256="b" * 64, sidecar=path)
    changed = model()
    with torch.no_grad():
        changed.policy.core.weight.add_(1)
    with pytest.raises(ValueError, match="incompatible"):
        attach_budget_head(changed, base_sha256="a" * 64, sidecar=path)


def dataset(path, lower, seed, episode_seeds):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(64, 100)).astype(np.float32)
    y = np.c_[x[:, :4] * .1, np.ones((64, 4), np.float32)].astype(np.float32)
    meta = dict(contract=SIDECAR_CONTRACT, action_digest=action_model_digest(lower),
                budget_config=budget_manifest(), episode_seeds=episode_seeds, audit_rows=len(x))
    np.savez(path, features=x, targets=y, metadata=np.asarray(json.dumps(meta)))


def test_budget_only_fit_reduces_validation_error_without_moving_action_model(tmp_path):
    item = model()
    lower = attach_budget_head(item, base_sha256="a" * 64)
    before = frozen_policy_digest(item.policy)
    train, val = tmp_path / "train.npz", tmp_path / "validation.npz"
    dataset(train, lower, 10, [10, 11])
    dataset(val, lower, 20, [20, 21])
    result = fit_budget_head(item, train, val, BudgetFitConfig(epochs=40, patience=8, learning_rate=.01))
    assert result["head_changed"] and result["best_epoch"] > 0
    assert result["best_validation_loss"] < result["initial_validation_loss"] * .5
    assert frozen_policy_digest(item.policy) == before
    assert all(not p.requires_grad for p in item.policy.parameters())
    dataset(val, lower, 20, [10, 21])
    with pytest.raises(ValueError, match="disjoint"):
        fit_budget_head(item, train, val)
    with pytest.raises(ValueError, match="separate"):
        fit_budget_head(item, train, train)


def test_frozen_cost_price_changes_the_tradeoff_and_rejects_invalid_values():
    prediction = np.r_[[.2, .3, .4, .5], np.log1p([10, 1000, 2000, 10000])]
    assert choose_budget(prediction, 1, training=False, round_index=10, cost_price=0) == 4
    assert choose_budget(prediction, 1, training=False, round_index=10) == 1
    assert choose_budget(prediction, 1, training=False, round_index=10, cost_price=1) == 0
    for price in [-1, np.inf, np.nan]:
        with pytest.raises(ValueError):
            choose_budget(prediction, 1, training=False, round_index=10, cost_price=price)
