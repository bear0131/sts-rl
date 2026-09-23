from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import rl.training.distributed_sb3 as distributed


class _NoReplayDistributedModel:
    def __init__(self) -> None:
        self.nosl_is_az = True
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
    ):
        assert not step_optimizer
        assert not prepare_replay
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
    timing_fields = (
        "learner_total_sec",
        "learner_replay_sample_sec",
        "learner_materialize_sec",
        "learner_cpu_metrics_sec",
        "learner_h2d_sec",
        "learner_forward_sec",
        "learner_backward_sec",
        "learner_gradient_sync_sec",
        "learner_optimizer_sec",
        "learner_unattributed_sec",
    )
    for name in timing_fields:
        assert np.isfinite(result[name]), name
        assert result[name] >= 0.0, name
    assert result["learner_total_sec"] >= result["learner_unattributed_sec"]
