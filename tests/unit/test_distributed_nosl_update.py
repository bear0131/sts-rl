from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import rl.training.distributed_sb3 as distributed


class _NoReplayDistributedModel:
    def __init__(self) -> None:
        self.nosl_is_az = True
        self.num_timesteps = 0
        self.nosl_is_az_inner_only = True
        self.nosl_is_az_updates = 3
        self.nosl_is_az_replay_enabled = False
        self._nosl_is_az_fresh_samples = [object(), object()]
        self._nosl_is_az_rollout_timings = {}
        self.device = torch.device("cpu")
        self.max_grad_norm = 1.0
        self.parameter = torch.nn.Parameter(torch.tensor(0.0))
        optimizer = torch.optim.SGD([self.parameter], lr=0.1)
        self.policy = SimpleNamespace(
            optimizer=optimizer,
            parameters=lambda: [self.parameter],
        )
        self.prepare_calls = 0
        self.train_consume_flags = []

    def _prepare_nosl_is_az_replay(self):
        self.prepare_calls += 1
        return {}

    def _nosl_is_az_replay_train_count(self):
        return len(self._nosl_is_az_fresh_samples)

    def _train_nosl_is_az(
        self,
        *,
        step_optimizer,
        gradient_scale,
        prepare_replay,
        consume_fresh_samples,
        update_count,
        value_only_updates,
    ):
        assert not step_optimizer
        assert not prepare_replay
        assert update_count == 1
        assert value_only_updates == 0
        self.train_consume_flags.append(consume_fresh_samples)
        update_index = len(self.train_consume_flags) - 1
        mean = (0.2, 0.5, 0.8)[update_index]
        std = 0.1
        count = float(len(self._nosl_is_az_fresh_samples))
        self.parameter.grad = torch.ones_like(self.parameter) * gradient_scale

        names = (
            "target_entropy", "policy_entropy", "target_policy_kl",
            "policy_cross_entropy", "target_effective_actions",
            "target_top1_agreement", "legal_actions",
            "illegal_target_mass", "target_sum_error", "value_target",
            "pred_value", "value_mae", "value_bias", "finite_fraction",
        )
        self._nosl_is_az_metric_sums = {name: 0.0 for name in names}
        self._nosl_is_az_metric_sumsq = {name: 0.0 for name in names}
        self._nosl_is_az_metric_counts = {name: count for name in names}
        for name in ("value_target", "pred_value"):
            self._nosl_is_az_metric_sums[name] = mean * count
            self._nosl_is_az_metric_sumsq[name] = (
                std * std + mean * mean
            ) * count
        self._nosl_is_az_metric_count = count
        self._nosl_is_az_policy_loss_sum = count
        self._nosl_is_az_policy_weight_sum = count
        self._nosl_is_az_value_loss_sum = count
        return {
            "roots": count,
            "policy_loss": 1.0,
            "value_loss": 1.0,
            "gradient_batches": 1.0,
            "metric_count": count,
            "value_target_min": mean - std,
            "value_target_max": mean + std,
            "fresh_roots": count,
            "batch_size_requested": count,
            "batch_size_used": count,
            "batch_fill_ratio": 1.0,
            "replay_sampled_roots": 0.0,
            "replay_old_fraction": 0.0,
            "replay_enabled": 0.0,
            "replay_size": 0.0,
        }


def test_distributed_no_replay_reuses_fresh_batch_for_every_update(monkeypatch):
    model = _NoReplayDistributedModel()
    monkeypatch.setattr(
        distributed,
        "average_scalar_dict",
        lambda values, _transport: dict(values),
    )
    monkeypatch.setattr(
        distributed,
        "synchronize_gradient_step",
        lambda *_args, **_kwargs: {
            "sync_sec": 0.0,
            "sync_mbps": 0.0,
            "grad_bytes": 4,
        },
    )

    result = distributed.distributed_nosl_is_az_update(
        model,
        rank=0,
        world_size=1,
        peers=[],
        key=Path("unused"),
        control_dir=Path("unused"),
        local_root=Path("unused"),
        remote_root="unused",
        run_id="test",
        update=0,
        ring_transport=None,
        barrier_timeout=1.0,
    )

    assert model.prepare_calls == 1
    assert model.train_consume_flags == [False, False, False]
    assert model._nosl_is_az_fresh_samples == []
    assert result["updates"] == 3.0
    assert result["roots"] == 6.0
    assert result["batch_size_used"] == 2.0
    assert result["value_target_mean"] == pytest.approx(0.5)
    # Mixture variance = within-update 0.1^2 + variance([0.2, 0.5, 0.8]).
    assert result["value_target_std"] == pytest.approx(np.sqrt(0.07))
    assert result["value_target_min"] == pytest.approx(0.1)
    assert result["value_target_max"] == pytest.approx(0.9)


def test_decreasing_update_budget_does_not_reuse_transport_tags(monkeypatch):
    tags = []
    def synchronize(*_args, **kwargs):
        tags.append(kwargs['tag'])
        return dict(sync_sec=0., sync_mbps=0., grad_bytes=4)
    monkeypatch.setattr(distributed, 'synchronize_gradient_step', synchronize)
    for rollout, budget in ((3, 3), (4, 2)):
        model = _NoReplayDistributedModel()
        distributed.distributed_nosl_is_az_update(model, rank=0, world_size=1, peers=[],
            key=Path('unused'),control_dir=Path('unused'),local_root=Path('unused'),
            remote_root='unused',run_id='test',update=rollout,update_count=budget,
            ring_transport=None,barrier_timeout=1)
    assert len(tags) == len(set(tags)) == 5


def test_nonfinite_synchronized_gradient_skips_adam_without_poisoning(monkeypatch):
    model = _NoReplayDistributedModel()
    model.nosl_is_az_updates = 1

    def synchronize(*_args, **_kwargs):
        model.parameter.grad = torch.tensor(float('nan'))
        return dict(sync_sec=0., sync_mbps=0., grad_bytes=4,
                    local_nonfinite_gradients=1.,
                    synchronized_nonfinite_gradients=1.)

    monkeypatch.setattr(distributed, 'synchronize_gradient_step', synchronize)
    before = model.parameter.detach().clone()
    result = distributed.distributed_nosl_is_az_update(
        model, rank=0, world_size=1, peers=[], key=Path('unused'),
        control_dir=Path('unused'), local_root=Path('unused'),
        remote_root='unused', run_id='test', update=0,
        ring_transport=None, barrier_timeout=1,
    )
    assert result['updates'] == 0
    assert result['skipped_nonfinite_updates'] == 1
    assert result['synchronized_nonfinite_gradients'] == 1
    assert model._nosl_is_az_skipped_nonfinite_updates == 1
    assert model.parameter.grad is None
    assert torch.equal(model.parameter, before)
    assert not model.policy.optimizer.state


def test_gradient_transport_detects_nonfinite_average_before_adam(tmp_path):
    model = _NoReplayDistributedModel()
    model.parameter.grad = torch.ones_like(model.parameter)

    class BadRing:
        def average(self, local):
            return torch.full_like(local, float('nan')), local.numel() * 4

    result = distributed.synchronize_gradient_step(
        model, rank=0, world_size=2, peers=[], key=tmp_path / 'key',
        control_dir=tmp_path, local_root=tmp_path,
        remote_root='unused', run_id='test', tag='bad',
        ring_transport=BadRing(), barrier_timeout=1,
    )
    assert result['local_nonfinite_gradients'] == 0
    assert result['synchronized_nonfinite_gradients'] == 1
    assert torch.isnan(model.parameter.grad)


def test_nonfinite_diagnostic_saves_exact_batch_and_pre_step_weights(tmp_path):
    policy = torch.nn.Linear(2, 1)
    policy.optimizer = torch.optim.Adam(policy.parameters(), lr=0.01)
    batch = {'sample_ids': np.array([17, 23], dtype=np.uint64),
             'value_targets': np.array([0.5, 0.0], dtype=np.float32)}
    model = SimpleNamespace(
        num_timesteps=49152, policy=policy,
        _nosl_is_az_diagnostic_batch=batch,
        _nosl_is_az_diagnostic_rng=np.random.get_state(),
        _nosl_is_az_diagnostic_fresh_start_id=17,
    )
    before = policy.weight.detach().clone()
    path = distributed.save_nonfinite_nosl_diagnostic(
        model, local_root=tmp_path, update=1, inner_update=2, rank=0,
        local_nonfinite_gradients=1, first_nonfinite_parameter='weight',
    )
    saved = torch.load(path, map_location='cpu', weights_only=False)
    assert saved['fresh_start_id'] == 17
    assert saved['first_nonfinite_parameter'] == 'weight'
    np.testing.assert_array_equal(saved['batch']['sample_ids'], batch['sample_ids'])
    assert torch.equal(saved['policy']['weight'], before)


def test_optimizer_rejects_nonfinite_norm_before_mutating_parameters():
    from rl.algorithms.updates import optimizer_step

    model = _NoReplayDistributedModel()
    model.parameter.grad = torch.tensor(float('nan'))
    before = model.parameter.detach().clone()
    with pytest.raises(RuntimeError, match='non-finite'):
        optimizer_step(model)
    assert torch.equal(model.parameter, before)
    assert not model.policy.optimizer.state
