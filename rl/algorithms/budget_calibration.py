"""Fit a budget head on frozen mature policies using independent audit targets.

Model tensors stay in the original checkpoint. A small sidecar is tied to its
exact SHA and can only replace the budget head, never the action model.
"""
from dataclasses import dataclass, asdict
from pathlib import Path
from types import SimpleNamespace
import copy
import json

import numpy as np
import torch

from rl.algorithms.updates import optimizer_step
from rl.algorithms.upper_probe import state_digest
from rl.policies.search_budget import AdaptiveTokenRegretPolicy, BUDGET_CONTRACT, budget_manifest
from rl.policies.token_regret import OnlineTokenRegretPolicy, pack_rows

SIDECAR_CONTRACT = "sts-frozen-policy-budget-head-v1"


def frozen_policy_digest(policy):
    return state_digest({k: v for k, v in policy.state_dict().items()
                         if not k.startswith("small_playout.budget_head.")
                         and k != "small_playout.budget_round"})


def action_model_digest(lower):
    return state_digest({k: v for k, v in lower.state_dict().items()
                         if not k.startswith("budget_head.") and k != "budget_round"})


def attach_budget_head(model, *, base_sha256, seed=20261108, sidecar=None):
    lower = model.policy.small_playout
    if not isinstance(lower, OnlineTokenRegretPolicy) or isinstance(lower, AdaptiveTokenRegretPolicy):
        raise ValueError("budget calibration requires an original token-root checkpoint")
    if len(base_sha256) != 64:
        raise ValueError("an exact base-checkpoint SHA256 is required")
    before = frozen_policy_digest(model.policy)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        adaptive = AdaptiveTokenRegretPolicy(lower.keys.num_embeddings)
    adaptive.to(device=next(lower.parameters()).device, dtype=next(lower.parameters()).dtype).eval()
    loaded = adaptive.load_state_dict(lower.state_dict(), strict=False)
    expected = {"budget_round", *("budget_head." + k for k in adaptive.budget_head.state_dict())}
    if set(loaded.missing_keys) != expected or loaded.unexpected_keys:
        raise ValueError("mature action-model parameters did not map exactly")
    if sidecar is not None:
        saved = torch.load(sidecar, map_location="cpu", weights_only=True)
        if (saved["contract"] != SIDECAR_CONTRACT or saved["base_sha256"] != base_sha256
                or saved["budget_config"] != budget_manifest() or saved["frozen_digest"] != before):
            raise ValueError("budget head is incompatible with the base checkpoint or budget contract")
        adaptive.budget_head.load_state_dict(saved["head"], strict=True)
        adaptive.budget_round.fill_(int(saved["budget_round"]))
    model.policy.small_playout = adaptive
    model.policy.requires_grad_(False)
    model.small_playout_search_contract = BUDGET_CONTRACT
    assert frozen_policy_digest(model.policy) == before
    model.budget_calibration_identity = dict(base_sha256=base_sha256, frozen_digest=before,
                                             head_digest=state_digest(adaptive.budget_head.state_dict()),
                                             budget_round=int(adaptive.budget_round))
    return adaptive


def save_budget_head(model, path, *, report=None):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    identity = model.budget_calibration_identity
    assert frozen_policy_digest(model.policy) == identity["frozen_digest"]
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(contract=SIDECAR_CONTRACT, base_sha256=identity["base_sha256"],
                    frozen_digest=identity["frozen_digest"],
                    budget_config=budget_manifest(),
                    budget_round=int(model.policy.small_playout.budget_round),
                    head={k: v.detach().cpu().clone() for k, v in
                          model.policy.small_playout.budget_head.state_dict().items()},
                    report=report), path)


class BudgetSampleCollector:
    """Keep only audited 100-feature rows; action labels are never optimized."""
    def __init__(self, path, lower):
        self.path = Path(path)
        if self.path.exists():
            raise FileExistsError(self.path)
        self.action_digest = action_model_digest(lower)
        self.features, self.targets = [], []
        self.seen_roots = 0
        self.episode_seeds = set()
        self.saved = False

    @torch.inference_mode()
    def add(self, lower, samples, *, episode_seeds=()):
        self.episode_seeds.update(int(seed) for seed in episode_seeds)
        self.seen_roots += len(samples)
        rows = [row for row in samples if row["budget_audit"]]
        if not rows:
            return
        device = next(lower.parameters()).device
        inputs = pack_rows(rows, device)
        scores, _, context = lower.scores_and_guide(*inputs, return_context=True)
        boss = torch.tensor([r["budget_boss"] for r in rows], device=device)
        features = lower.budget_features(context, inputs[0], scores, inputs[-1], boss)
        targets = np.asarray([r["budget_targets"] for r in rows], dtype=np.float32)
        assert features.shape == (len(rows), 100) and targets.shape == (len(rows), 8)
        assert torch.isfinite(features).all() and np.isfinite(targets).all()
        self.features.append(features.cpu().numpy().copy())
        self.targets.append(targets.copy())

    def save(self, lower):
        if self.saved:
            return
        assert action_model_digest(lower) == self.action_digest
        features = np.concatenate(self.features) if self.features else np.empty((0, 100), np.float32)
        targets = np.concatenate(self.targets) if self.targets else np.empty((0, 8), np.float32)
        metadata = dict(contract=SIDECAR_CONTRACT, action_digest=self.action_digest,
                        budget_config=budget_manifest(), seen_roots=self.seen_roots,
                        audit_rows=len(features), episode_seeds=sorted(self.episode_seeds))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("xb") as stream:
            np.savez_compressed(stream, features=features, targets=targets,
                                metadata=np.asarray(json.dumps(metadata)))
        self.saved = True


@dataclass(frozen=True)
class BudgetFitConfig:
    seed: int = 20261108
    epochs: int = 200
    patience: int = 20
    batch_size: int = 256
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    cost_loss_weight: float = .1
    max_grad_norm: float = 1.

    def __post_init__(self):
        if min(self.epochs, self.patience, self.batch_size) <= 0:
            raise ValueError("positive fitting budgets required")
        if not all(np.isfinite(x) and x >= 0 for x in
                   (self.learning_rate, self.weight_decay, self.cost_loss_weight, self.max_grad_norm)):
            raise ValueError("invalid fitting hyperparameters")


def fit_budget_head(model, train_path, validation_path, config=BudgetFitConfig()):
    lower = model.policy.small_playout
    if not isinstance(lower, AdaptiveTokenRegretPolicy):
        raise ValueError("attach the new head before fitting")
    if Path(train_path).resolve() == Path(validation_path).resolve():
        raise ValueError("validation must be a separate collection")
    original = frozen_policy_digest(model.policy)
    before = state_digest(lower.budget_head.state_dict())
    device = next(lower.parameters()).device
    datasets, seed_sets = [], []
    for path in (train_path, validation_path):
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(str(data["metadata"]))
            if (meta["contract"] != SIDECAR_CONTRACT or meta["budget_config"] != budget_manifest()
                    or meta["action_digest"] != action_model_digest(lower)):
                raise ValueError("audit dataset uses a different frozen action model")
            x, y = data["features"].copy(), data["targets"].copy()
            seed_sets.append(set(meta["episode_seeds"]))
        if len(x) < 16 or x.shape != (len(y), 100) or y.shape[1:] != (8,):
            raise ValueError("insufficient or malformed audit rows")
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError("nonfinite audit data")
        datasets.append((torch.as_tensor(x, device=device), torch.as_tensor(y, device=device)))
    if not all(seed_sets) or seed_sets[0] & seed_sets[1]:
        raise ValueError("training and validation must use disjoint actual episode seeds")
    model.policy.requires_grad_(False)
    lower.budget_head.requires_grad_(True)
    parameters = list(lower.budget_head.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=config.learning_rate, weight_decay=config.weight_decay)
    proxy = SimpleNamespace(policy=lower.budget_head, max_grad_norm=config.max_grad_norm)
    generator = torch.Generator().manual_seed(config.seed)

    def loss(x, y):
        squared = (lower.budget_head(x) - y).square()
        return squared[:, :4].mean() + config.cost_loss_weight * squared[:, 4:].mean()

    def validation():
        with torch.no_grad():
            x, y = datasets[1]
            value = loss(x, y)
            assert torch.isfinite(value)
            return float(value)

    best_loss = initial_validation_loss = validation()
    best_state = copy.deepcopy(lower.budget_head.state_dict())
    best_epoch, stale, history = 0, 0, []
    try:
        for epoch in range(1, config.epochs + 1):
            x, y = datasets[0]
            order = torch.randperm(len(x), generator=generator)
            total = 0.
            for begin in range(0, len(x), config.batch_size):
                indices = order[begin:begin + config.batch_size].to(device)
                value = loss(x[indices], y[indices])
                if not torch.isfinite(value):
                    raise FloatingPointError("nonfinite budget-only loss")
                value.backward()
                optimizer_step(proxy, optimizer=optimizer)
                total += float(value.detach()) * len(indices)
            current = validation()
            history.append(dict(epoch=epoch, train_loss=total / len(x), validation_loss=current))
            if current < best_loss:
                best_loss, best_epoch = current, epoch
                best_state, stale = copy.deepcopy(lower.budget_head.state_dict()), 0
            else:
                stale += 1
            if stale >= config.patience:
                break
        lower.budget_head.load_state_dict(best_state)
        lower.budget_round.fill_(3)
    finally:
        model.policy.requires_grad_(False)
    assert frozen_policy_digest(model.policy) == original
    return dict(config=asdict(config), train_rows=len(datasets[0][0]), validation_rows=len(datasets[1][0]),
                best_epoch=best_epoch, best_validation_loss=best_loss, initial_validation_loss=initial_validation_loss, history=history,
                head_changed=state_digest(lower.budget_head.state_dict()) != before,
                frozen_weights_unchanged=True)
