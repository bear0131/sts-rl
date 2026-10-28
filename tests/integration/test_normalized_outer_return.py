"""Remaining-reward units across real search settlement and executed labels."""
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from env.core import encode, vocab
from env.core.outer_return import (CombatExit, NORMALIZED_OBJECTIVE, normalized_exit,
                                   remaining_reward_max)
from env.core.run_env import SlayTheSpireEnv
from rl.algorithms import outer_return as outer

pytestmark = [pytest.mark.native, pytest.mark.timeout(30)]


@pytest.mark.parametrize('act,floor,expected', [(1, 0, 1.402), (1, 16, 1.37),
                                               (2, 17, 1.268), (3, 34, 1.034), (3, 50, 1.002)])
def test_remaining_reward_ceiling(act, floor, expected):
    assert remaining_reward_max(act, floor) == pytest.approx(expected)


def test_settlement_reward_and_continuation_have_source_units():
    # Act completion consumes both one floor and the 0.1 Act1 bonus.
    assert normalized_exit(CombatExit(.102, 'upper'), 1, 16, 2, 17).target(1.268) == pytest.approx(1.)
    # A next combat predicts its own normalized value, not a raw teacher value.
    assert normalized_exit(CombatExit(.102, 'combat'), 1, 16, 2, 17).target(.5) == pytest.approx((.102 + .634)/1.37)
    assert normalized_exit(CombatExit(1.002, 'terminal'), 3, 50, 3, 51).target() == pytest.approx(1.)
    assert normalized_exit(CombatExit(0., 'terminal'), 1, 4, 1, 4).target() == 0.
    assert remaining_reward_max(4, 51, terminal=True) == 0.
    with pytest.raises(ValueError, match='live'):
        remaining_reward_max(4, 51)
    # Learned extrapolation is observed, not silently clipped.
    assert normalized_exit(CombatExit(0., 'upper'), 3, 50, 3, 50).target(2.) > 1.


def test_executed_targets_match_boundary_units_and_survive_pending_collection():
    class Teacher:
        def assert_integrity(self):
            pass
        def __call__(self, obs):
            return None, torch.full((len(obs['phase']), 1), 1.268)
    model = SimpleNamespace(policy=SimpleNamespace(core=SimpleNamespace(teacher=Teacher(),
        training_objective=NORMALIZED_OBJECTIVE,
        combat_actor_critic=lambda obs: (None, torch.full((len(obs['phase']), 1), .5)))),
        device='cpu', autocast_context=nullcontext, _nosl_is_az_samples=[],
        _nosl_is_az_pending_fights=[[] for _ in range(4)])
    pending = model._nosl_is_az_pending_fights
    before = dict(phase=np.full((4, 1), vocab.PHASE_VOCAB_INDEX['combat']),
                  act=np.full((4, 1), 1/encode.ACT_SCALE), floor=np.full((4, 1), 16/encode.FLOOR_SCALE))
    model._last_obs = before
    outer.begin_collection(model, 4)
    for row in pending:
        row.append({})
    outer.stamp_roots(model, pending)
    outer.begin_collection(model, 4)  # unfinished roots keep their source divisor
    after = dict(phase=before['phase'].copy(), act=np.full((4, 1), 2/encode.ACT_SCALE),
                 floor=np.full((4, 1), 17/encode.FLOOR_SCALE))
    after['phase'][0] = vocab.PHASE_VOCAB_INDEX['map']
    stats = dict(act=2, floor=17, outcome='UNDECIDED')
    infos = [dict(run_stats=stats), dict(run_stats=stats, outer_combat_exit=True),
             dict(run_stats=dict(act=1, floor=16, outcome='PLAYER_LOSS')),
             dict(run_stats=stats, **{'TimeLimit.truncated': True})]
    assert outer.finish_step(model, before, after, [False, False, True, True], infos, pending) == (3, 1, 3)
    assert [s['value_target'] for s in model._nosl_is_az_samples] == pytest.approx([1., (.102+.634)/1.37, 0.])
    assert model._outer_collection['collector_upper_rows'] == 1
    assert model._outer_collection['collector_upper_value_seconds'] > 0.
    pending[0].append(dict(outer_reward_prefix=0.))
    with pytest.raises(ValueError, match='pending scale'):
        outer.validate_pending(model, 4)


def test_real_search_exit_divisor_and_checkpoint_objective_guard(run_combat_factory):
    import slaythespire as sts
    backend = run_combat_factory(potions=['Smoke Bomb'])
    env = SlayTheSpireEnv(outer_return_context=True, combat_action_contract='autoregressive-v2',
                         combat_value_objective=NORMALIZED_OBJECTIVE)
    try:
        env.reset(seed=42101)
        env._nosl_outer_branch = backend.fork_run()
        leaf = backend.bc.clone()
        leaf.execute(sts.Action(sts.ActionType.POTION, 0, 0))
        outcome, obs = env.settle_search_combat_exit(leaf)
        assert outcome.kind == 'upper' and obs is not None
        assert outcome.target(.7) == pytest.approx(.7 / remaining_reward_max(backend.gc.act, backend.gc.floor_num))
        state = env.export_training_state()
        state['outer_return_normalized'] = False
        with pytest.raises(ValueError, match='normalization mismatch'):
            env.restore_training_state(state)
    finally:
        env.close()


def test_qnorm_and_unsupported_ascension_rejected():
    model = SimpleNamespace(policy=SimpleNamespace(core=SimpleNamespace(training_objective=NORMALIZED_OBJECTIVE)),
                            nosl_is_az_gumbel_rescale_q=True)
    with pytest.raises(ValueError, match='qnorm'):
        outer.setup(model)
    with pytest.raises(ValueError, match='A0'):
        SlayTheSpireEnv(outer_return_context=True, combat_action_contract='autoregressive-v2',
                       combat_value_objective=NORMALIZED_OBJECTIVE, ascension=1)
    with pytest.raises(ValueError, match='public run context'):
        SlayTheSpireEnv(combat_value_objective=NORMALIZED_OBJECTIVE)
