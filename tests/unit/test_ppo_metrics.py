import numpy as np
import pytest

from env.core import vocab
from rl.algorithms.ppo_metrics import value_moments, finalize_metrics


def test_value_metrics_reduce_moments_with_unequal_rank_and_phase_counts():
    combat, upper = (vocab.PHASE_VOCAB_INDEX[k] for k in ('combat', 'map'))
    values = np.array([0., 2., 1., 3., 7.])
    targets = np.array([1., 3., 4., 4., 6.])
    phases = np.array([combat, upper, combat, combat, combat])
    a = value_moments(values[:1], targets[:1], phases[:1])
    b = value_moments(values[1:], targets[1:], phases[1:])
    result = finalize_metrics({key: (a[key] + b[key]) / 2 for key in a})
    for name, mask in [('all', np.ones(5, bool)), ('combat', phases == combat), ('upper', phases == upper)]:
        p = f'pure_rl/value_pre/{name}/'
        errors = values[mask] - targets[mask]
        assert result[p+'mse'] == pytest.approx(np.square(errors).mean())
        assert result[p+'bias'] == pytest.approx(errors.mean())
        if np.var(targets[mask]) > 0:
            assert result[p+'explained_variance'] == pytest.approx(1 - np.var(errors) / np.var(targets[mask]))
        else:
            assert p+'explained_variance' not in result


def test_empty_phase_is_missing_not_zero_and_targets_are_not_mutated():
    values, targets = np.zeros((2, 1)), np.ones((2, 1))
    phase = np.full((2, 1), vocab.PHASE_VOCAB_INDEX['combat'])
    result = finalize_metrics(value_moments(values, targets, phase))
    assert result['pure_rl/value_pre/all/mse'] == 1
    assert not any('/upper/' in key or 'explained_variance' in key for key in result)
    np.testing.assert_array_equal(values, np.zeros((2, 1)))
    np.testing.assert_array_equal(targets, np.ones((2, 1)))
    with pytest.raises(ValueError, match='same shape'):
        value_moments(values, targets.flatten(), phase)


def test_update_losses_are_weighted_by_samples_not_rank_or_microbatch():
    from types import SimpleNamespace
    import torch
    from rl.algorithms.ppo_metrics import accumulate_update_metrics, update_moments
    model = SimpleNamespace()
    for n, loss in [(3, 2.), (1, 10.)]:
        x = torch.tensor(loss, requires_grad=True)
        c = SimpleNamespace(policy_loss=x, value_loss=x, entropy_loss=-x,
                            approx_kl=x*.01, clip_fraction=x*.02)
        accumulate_update_metrics(model, c, n)
        assert x.grad is None
    moments = update_moments(model)
    result = finalize_metrics(moments)
    assert moments['pure_rl/update/count'] == 4
    assert result['pure_rl/value_loss'] == result['pure_rl/policy_loss'] == result['pure_rl/entropy'] == 4
    assert result['pure_rl/approx_kl'] == pytest.approx(.04)
    assert result['pure_rl/clip_fraction'] == pytest.approx(.08)
