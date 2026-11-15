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


def test_teacher_top1_uses_global_searched_root_counts_and_keeps_missing_missing():
    p = 'deck_search/teacher_top1'
    # One perfect rank, one larger rank with no matches, and two empty ranks.
    result = finalize_metrics({p + '_matches': .25, p + '_count': 1.}, world_size=4)
    assert result[p] == .25  # 1 / (1 + 3), not the mean of rank accuracies.
    assert result[p + '_samples_global'] == 4
    empty = finalize_metrics({p + '_matches': 0., p + '_count': 0.}, world_size=4)
    assert p not in empty and empty[p + '_samples_global'] == 0
    assert not any(key.startswith(p) for key in finalize_metrics({}))


def test_policy_entropy_keeps_searched_rows_but_excludes_value_only_anchors():
    import torch
    from rl.algorithms.ppo import masked_categorical_statistics
    from rl.algorithms.ppo_metrics import policy_entropy_moments
    combat, upper = (vocab.PHASE_VOCAB_INDEX[k] for k in ('combat', 'map'))
    logits = torch.zeros(5, 4, requires_grad=True)
    masks = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0], [1, 0, 0, 0],
                          [1, 1, 1, 1], [1, 1, 0, 0]], dtype=torch.bool)
    _, entropy, _ = masked_categorical_statistics(logits, torch.zeros(5, dtype=torch.long), masks)
    obs = dict(phase=torch.tensor([upper, combat, combat, combat, upper]),
               _search_executed=torch.tensor([0, 1, 0, 0, 0]),
               combat_exit_anchor=torch.tensor([[0], [0], [0], [1], [0]]))
    moments = policy_entropy_moments(entropy, obs)
    torch.testing.assert_close(moments, torch.tensor([np.log(8), 2., np.log(2), 2.], dtype=torch.float64))
    assert not moments.requires_grad and logits.grad is None


def test_policy_entropy_uses_each_layer_count_after_unequal_rank_reduction():
    from types import SimpleNamespace
    import torch
    from rl.algorithms.ppo_metrics import accumulate_update_metrics, update_moments
    reports = []
    for moments in ([1., 1., 4., 2.], [9., 3., 0., 0.]):
        model = SimpleNamespace()
        zero = torch.tensor(0.)
        components = SimpleNamespace(policy_loss=zero, value_loss=zero, entropy_loss=zero,
            approx_kl=zero, clip_fraction=zero, policy_entropy_moments=torch.tensor(moments).double())
        accumulate_update_metrics(model, components, 4)
        reports.append(update_moments(model))
    result = finalize_metrics({k: sum(r[k] for r in reports)/2 for k in reports[0]}, world_size=2)
    assert result['pure_rl/entropy_upper'] == 2.5
    assert result['pure_rl/entropy_lower'] == 2.
    assert result['pure_rl/entropy_upper_samples_global'] == 4
    assert result['pure_rl/entropy_lower_samples_global'] == 2
    empty = finalize_metrics(update_moments(SimpleNamespace()))
    assert 'pure_rl/entropy_lower' not in empty and 'pure_rl/entropy_upper' not in empty
    assert not any('entropy_lower' in k for k in finalize_metrics({}))


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_layer_diagnostics_preserve_actual_ppo_loss_and_gradients(monkeypatch, device):
    from contextlib import nullcontext
    from types import SimpleNamespace
    import torch
    from gymnasium import spaces
    from sb3_contrib.common.maskable.distributions import MaskableCategorical
    from rl.algorithms import ppo_metrics
    from rl.algorithms.updates import ppo_forward
    if device == 'cuda':
        assert torch.cuda.is_available(), 'explicit CUDA validation is required'
    combat, upper = (vocab.PHASE_VOCAB_INDEX[k] for k in ('combat', 'map'))
    original = ppo_metrics.policy_entropy_moments
    outcomes = []
    for enabled in (False, True):
        x = torch.arange(20, device=device, dtype=torch.float32).reshape(4, 5).div(11).requires_grad_()
        def evaluate(obs, actions, action_masks):
            distribution = MaskableCategorical(logits=x)
            distribution.apply_masking(action_masks)
            return x[:, 0], distribution.log_prob(actions), distribution.entropy()
        model = SimpleNamespace(policy=SimpleNamespace(evaluate_actions=evaluate),
            action_space=spaces.Discrete(5), autocast_context=nullcontext,
            normalize_advantage=False, hierarchical_combat_hp=False, ent_coef=.01, vf_coef=.5)
        batch = SimpleNamespace(observations=dict(phase=torch.tensor([combat, combat, upper, combat],device=device),
            _search_executed=torch.tensor([1, 0, 0, 0],device=device),
            combat_exit_anchor=torch.tensor([[0], [0], [0], [1]],device=device)),
            actions=torch.tensor([1, 2, 3, 0],device=device), action_masks=torch.ones(4,5,device=device),
            advantages=torch.tensor([1., -.5, .25, 0.],device=device), returns=torch.ones(4,device=device),
            old_values=torch.zeros(4,device=device), old_log_prob=torch.full((4,),-1.5,device=device))
        monkeypatch.setattr(ppo_metrics, 'policy_entropy_moments', original if enabled else lambda *a: None)
        components, loss = ppo_forward(model,batch,.2,None)
        outcomes.append((loss.detach(), torch.autograd.grad(loss,x)[0]))
        if enabled:
            assert components.policy_entropy_moments.tolist()[1::2] == [1., 2.]
    for a,b in zip(*outcomes,strict=True):
        torch.testing.assert_close(a,b,atol=0,rtol=0)
