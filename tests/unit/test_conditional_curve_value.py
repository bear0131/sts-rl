"""Victory-factorized finite returns, including non-victories and >1 values."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from env.core.finite_curves import FACTORED_OBJECTIVE, winning_exit_target
from env.core.outer_return import CombatExit
from rl.algorithms.combat_distribution import conditional_value_loss, CONDITIONAL_METRICS
from rl.algorithms import outer_return as outer
from rl.policies.combat_distribution import CombatConditionalValueHead


def test_conditional_value_is_unbounded_and_product_is_not_an_extra_loss():
    head = CombatConditionalValueHead(8)
    with torch.no_grad():
        for parameter in head.parameters():
            parameter.zero_()
        head.value[-1].bias.fill_(2.5)
    value, outputs = head(torch.zeros(3, 8))
    torch.testing.assert_close(value, torch.full((3, 1), 1.25))
    targets = torch.zeros(3, 9)
    targets[0, 0] = 1
    loss, metrics = conditional_value_loss(outputs, targets, torch.tensor([3., 0., 0.]))
    torch.testing.assert_close(loss, torch.log(torch.tensor(2.)) + torch.tensor([.25, 0., 0.]))
    assert tuple(metrics) == CONDITIONAL_METRICS
    outputs['value_given_win'].retain_grad()
    outputs['win_logits'].retain_grad()
    loss.sum().backward()
    torch.testing.assert_close(outputs['value_given_win'].grad, torch.tensor([[-1.], [0.], [0.]]))
    torch.testing.assert_close(outputs['win_logits'].grad, torch.tensor([[-.5], [.5], [.5]]))


def test_search_keeps_the_product_for_bosses_too():
    from rl.search.native import NativeSearchConfig
    model = SimpleNamespace(nosl_is_az_fixed_worldline=True, policy=SimpleNamespace(entity_config={'boss_win_weight': .9},
        core=SimpleNamespace(training_objective=FACTORED_OBJECTIVE)))
    config = NativeSearchConfig.from_model(model)
    assert config.outer_return and config.boss_win_weight == 0


def test_win_curves_keep_baseline_return_while_escape_and_loss_are_zero():
    bank = SimpleNamespace(value=lambda state: 1.7)
    for kind in ('upper', 'combat', 'terminal'):
        win = CombatExit(.102, kind, dict(outlet='win'))
        assert winning_exit_target(win, bank, reward_prefix=.002) == pytest.approx(
            (1., .1 if kind == 'terminal' else 1.8))
        for outlet in ('escape', 'loss'):
            assert winning_exit_target(CombatExit(.102, kind, dict(outlet=outlet)), bank) == (0., 0.)
    with pytest.raises(ValueError, match='observed combat exit'):
        winning_exit_target(CombatExit(0., 'censored', dict(outlet='loss')), bank)


def test_executed_labels_keep_outcome_and_prefix_without_teacher_calls():
    from env.core import encode, vocab as V
    bank = SimpleNamespace(value=lambda state: 1.7)
    model = SimpleNamespace(policy=SimpleNamespace(core=SimpleNamespace(
        teacher=SimpleNamespace(assert_integrity=lambda: None), training_objective=FACTORED_OBJECTIVE,
        finite_curve_bank=bank)), _nosl_is_az_samples=[], _nosl_is_az_pending_fights=[[] for _ in range(4)])
    outer.begin_collection(model, 4)
    before = dict(phase=np.ones((4, 1)), act=np.full((4, 1), 1/encode.ACT_SCALE),
                  floor=np.full((4, 1), 3/encode.FLOOR_SCALE))
    after = dict(before, phase=np.full((4, 1), V.PHASE_VOCAB_INDEX['map']))
    pending = model._nosl_is_az_pending_fights
    for rows in pending:
        rows.append(dict(outer_reward_prefix=0.))
        rows.append(dict(outer_reward_prefix=.002))
    model._outer_pending_rewards[:] = .002
    infos = [dict(run_stats=dict(act=1, floor=3, outcome=o), finite_curve_exit=dict(outlet=o))
             for o in ('win', 'escape', 'loss', 'loss')]
    infos[3]['TimeLimit.truncated'] = True
    assert outer.finish_step(model, before, after, [False, False, True, True], infos, pending) == (6, 2, 3)
    samples = model._nosl_is_az_samples
    assert [s['value_target'] for s in samples] == pytest.approx([1.702, 1.7, 0., 0., 0., 0.])
    assert [s['combat_targets'][0] for s in samples] == [1, 1, 0, 0, 0, 0]
    assert [s['outer_outcome'] for s in samples] == ['win', 'win', 'escape', 'escape', 'loss', 'loss']
