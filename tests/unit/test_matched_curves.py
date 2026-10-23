"""The sole payoff change in the COMBAT06-matched experiment."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from env.core.finite_curves import matched_exit_value, MATCHED_VALUE_SCALE
from env.core.outer_return import CombatExit
from rl.algorithms.combat_distribution import mean_outcome_loss


def test_fixed_normalization_includes_reward_and_never_clips():
    bank = SimpleNamespace(value=lambda state: 1.3)
    win = dict(outlet='win')
    assert MATCHED_VALUE_SCALE == 2.
    assert matched_exit_value(CombatExit(.2, 'upper', win), bank) == pytest.approx(.75)
    assert matched_exit_value(CombatExit(1.002, 'terminal', win), bank) == pytest.approx(.501)
    assert matched_exit_value(CombatExit(0., 'terminal', dict(outlet='loss')), bank) == 0.
    assert matched_exit_value(CombatExit(0., 'upper', dict(outlet='escape', combat_hp=80)), bank) == .4
    with pytest.raises(ValueError, match='fixed sigmoid range'):
        matched_exit_value(CombatExit(.8, 'upper', win), bank)
    with pytest.raises(ValueError, match='observed'):
        matched_exit_value(CombatExit(0., 'censored'), bank)


def test_same_two_losses_and_no_conditional_gradient_on_death():
    logits = torch.tensor([.5, -.5], requires_grad=True)
    means = torch.tensor([.6, .9], requires_grad=True)
    target = torch.zeros(2, 9)
    target[0, 0] = 1.
    losses, metrics = mean_outcome_loss(dict(win_logits=logits, hp_mean=means), target, torch.tensor([.7, 0.]))
    torch.testing.assert_close(losses, metrics['win_bce'] + metrics['hp_mse'])
    losses.mean().backward()
    assert means.grad[0].item() == pytest.approx(-.1)
    assert means.grad[1].item() == 0.
    assert (logits.grad != 0).all()


def test_profile_changes_only_declared_payoff_contract():
    root = Path(__file__).resolve().parents[2] / 'configs/train'
    old = json.loads((root/'combat_two_means_876_4x5090.json').read_text())
    new = json.loads((root/'combat_matched_curves_876_4x5090.json').read_text())
    for key in new['strict_control']['required_equal_sections']:
        assert new[key] == old[key], key
    assert new['training']['seed'] == old['training']['seed'] == 9001000
    assert new['contract']['value_scale'] == 2.
