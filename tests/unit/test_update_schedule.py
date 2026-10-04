import pytest

from rl.algorithms.update_schedule import CombatUpdateSchedule


def test_value_warmup_curve_preserves_four_joint_updates():
    schedule = CombatUpdateSchedule(mode='value-warmup')
    plans = [schedule.plan(i, 4) for i in range(25)]
    assert plans[:5] == [(40, 36)] * 5
    assert plans[9] == (31, 27)
    assert plans[14] == (13, 9)
    assert plans[19:] == [(4, 0)] * 6
    assert all(total - extra == 4 for total, extra in plans)
    assert all(a[0] >= b[0] for a, b in zip(plans, plans[1:]))


def test_resume_reconstructs_clock_without_restarting_warmup():
    schedule = CombatUpdateSchedule(mode='value-warmup')
    restored = CombatUpdateSchedule.from_config(schedule.config())
    for global_step in (0, 245760, 491520, 983040, 50000000):
        assert restored.plan(global_step // 49152, 4) == schedule.plan(global_step // 49152, 4)
    assert restored.plan(491520 // 49152, 4)[0] < 40
    assert restored.plan(983040 // 49152, 4) == (4, 0)


@pytest.mark.parametrize('budget', [1, 4, 40])
def test_legacy_default_is_constant(budget):
    assert CombatUpdateSchedule.from_config().plan(0, budget) == (budget, 0)
    assert CombatUpdateSchedule.from_config().plan(10000, budget) == (budget, 0)


@pytest.mark.parametrize('config', [{'mode':'unknown'}, {'peak_updates':0},
                                    {'hold_rollouts':-1}, {'decay_rollouts':1.5}])
def test_rejects_invalid_schedule(config):
    with pytest.raises(ValueError):
        CombatUpdateSchedule.from_config(config)


def test_rejects_peak_below_joint_budget():
    with pytest.raises(ValueError, match='peak'):
        CombatUpdateSchedule(mode='value-warmup').plan(0, 41)


def test_step_changes_all_joint_updates_only_after_completed_round_100():
    schedule = CombatUpdateSchedule(mode='step', hold_rollouts=100, final_updates=16)
    assert [schedule.plan(i, 40) for i in range(100)] == [(40, 0)] * 100
    assert schedule.plan(100, 40) == (16, 0)
    assert schedule.plan(101, 40) == (16, 0)
    restored = CombatUpdateSchedule.from_config(schedule.config())
    for step, expected in ((99 * 49152, 40), (4915200, 16), (50_000_000, 16)):
        assert restored.plan(step // 49152, 40) == (expected, 0)


@pytest.mark.parametrize('value', [None, 0, -1, 1.5, True])
def test_step_rejects_missing_or_invalid_final_budget(value):
    with pytest.raises(ValueError, match='final_updates'):
        CombatUpdateSchedule(mode='step', final_updates=value)


def test_constant_rejects_silently_ignored_step_budget():
    with pytest.raises(ValueError, match='only supported'):
        CombatUpdateSchedule(final_updates=16)
