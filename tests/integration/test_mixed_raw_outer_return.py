"""Raw whole-fight labels and search-only conversion, including native leaves."""
from contextlib import nullcontext
from types import SimpleNamespace
import copy

import numpy as np
import pytest
import torch

from env.core import encode, vocab, native_actions as C
from env.core.outer_return import MIXED_RAW_OBJECTIVE, remaining_reward_max, CombatExit
from env.core.run_env import SlayTheSpireEnv
from env.search.native_batch import NativeSearchBatch
from rl.algorithms import outer_return as outer
from rl.search.native import forward_outer_leaves

pytestmark = [pytest.mark.native, pytest.mark.timeout(30)]


def test_whole_fight_shared_raw_target_across_collection_and_censor():
    class Teacher:
        def assert_integrity(self):
            pass

        def __call__(self, obs):
            return None, torch.full((len(obs['phase']), 1), 2.7)

    model = SimpleNamespace(policy=SimpleNamespace(core=SimpleNamespace(teacher=Teacher(),
        training_objective=MIXED_RAW_OBJECTIVE,
        combat_actor_critic=lambda obs: (None, torch.full((len(obs['phase']), 1), .8)))),
        device='cpu', autocast_context=nullcontext, _nosl_is_az_samples=[],
        _nosl_is_az_pending_fights=[[] for _ in range(4)])
    pending = model._nosl_is_az_pending_fights
    before = dict(phase=np.full((4, 1), vocab.PHASE_VOCAB_INDEX['combat']),
                  act=np.full((4, 1), 1/encode.ACT_SCALE), floor=np.full((4, 1), 15/encode.FLOOR_SCALE))
    outer.begin_collection(model, 4)
    for row in pending:
        row.append(dict(estimated_value=.2, search_value=99.))
    outer.stamp_roots(model, pending)
    progress = dict(run_stats=dict(act=1, floor=16, outcome='UNDECIDED'))
    assert outer.finish_step(model, before, before, [False]*4, [progress]*4, pending) == (0, 0, 0)
    # A restored pending list retains sampled estimates, not current predictions.
    model._nosl_is_az_pending_fights = pending = copy.deepcopy(pending)
    outer.begin_collection(model, 4)
    before['floor'][:] = 16/encode.FLOOR_SCALE
    for row in pending:
        row.append(dict(estimated_value=.6, search_value=-99.))
    outer.stamp_roots(model, pending)
    after = copy.deepcopy(before)
    after['phase'][0] = vocab.PHASE_VOCAB_INDEX['map']
    infos = [progress, dict(progress, outer_combat_exit=True),
             dict(run_stats=dict(act=1, floor=16, outcome='PLAYER_LOSS')),
             dict(progress, **{'TimeLimit.truncated': True})]
    assert outer.finish_step(model, before, after, [False, False, True, True], infos, pending) == (6, 2, 3)
    # Full fight shares .5*(exit including .002 reward)+.5*mean(.2,.6).
    assert [s['value_target'] for s in model._nosl_is_az_samples] == pytest.approx(
        [1.551, 1.551, .601, .601, .201, .201])
    assert not any(pending)
    assert model._outer_collection['mixed_fights'] == 3
    pending[0].append(dict(outer_reward_prefix=0., estimated_value=np.nan))
    with pytest.raises(ValueError, match='critic estimate'):
        outer.validate_pending(model, 4)


@pytest.mark.parametrize('direct', [False, True])
@pytest.mark.parametrize('kind', ['ordinary', 'victory', 'escape', 'loss'])
def test_native_search_raw_lower_and_upper_divide_once(run_combat_factory, kind, direct):
    backend = run_combat_factory(potions=['Smoke Bomb'] if kind == 'escape' else [])
    action = C.AUTOREGRESSIVE_PLAY
    if kind == 'victory':
        backend.bc.set_player_strength(999)
    elif kind == 'escape':
        action = C.AUTOREGRESSIVE_POTION + 5
    elif kind == 'loss':
        backend.bc.apply_custom_state({'player': {'hp': 1}})
        action = 0
    env = SlayTheSpireEnv(outer_return_context=True, combat_action_contract='autoregressive-v2',
                         combat_value_objective=MIXED_RAW_OBJECTIVE)
    batch = None
    try:
        env.reset(seed=42101)
        env.thick = backend
        env._invalidate_backend_cache()
        original = backend.bc.to_dict()
        scale = remaining_reward_max(backend.gc.act, backend.gc.floor_num)
        batch = NativeSearchBatch([env], 1, direct_observations=direct)
        logits = np.full(env.action_space.n, -1000.)
        logits[action] = 1000.
        battle, root_values = backend.bc, []
        class RecordNativeStart:
            def __getattr__(self, name):
                return getattr(battle, name)

            def nosl_neural_search_begin(self, *args):
                root_values.append(args[7])
                return battle.nosl_neural_search_begin(*args)

        backend.bc = RecordNativeStart()
        try:
            batch.begin({0: ((logits,), dict(simulations=1, depth=0, state_nonce=42101,
                root_value=2.7, fixed_worldline=True, outer_return=True, compact_observation=True))})
        finally:
            backend.bc = battle
        assert root_values == pytest.approx([2.7 / scale])
        assert list(batch.combat_exit_rows) == ([] if kind == 'ordinary' else [0])
        lower = lambda observations, width: (np.zeros((len(observations['relics']), env.action_space.n), np.float32),
                                            np.full(len(observations['relics']), 2.7, np.float32))
        teacher = lambda obs: (None, torch.full((len(obs['phase']), 1), 2.7))
        model = SimpleNamespace(policy=SimpleNamespace(core=SimpleNamespace(
            training_objective=MIXED_RAW_OBJECTIVE, teacher=teacher)), action_space=env.action_space,
            device='cpu')
        raw_exit = backend.settle_combat_leaf(batch.payload['combat_exits'][0])[1] if kind != 'ordinary' else None
        expected_raw = 2.7 if raw_exit is None else raw_exit.target(None if kind == 'loss' else 2.7)
        output_logits, values = forward_outer_leaves(model, batch, leaf_backend=lower)
        assert values == pytest.approx([expected_raw / scale])
        batch.advance(output_logits, values)
        assert batch.done
        assert batch.results()[0]['root_value'] == pytest.approx(expected_raw / scale)
        assert backend.bc.to_dict() == original
        state = env.export_training_state()
        state['outer_return_mixed_raw'] = False
        with pytest.raises(ValueError, match='mixed raw'):
            env.restore_training_state(state)
    finally:
        if batch is not None:
            batch.close()
        env.close()


def test_combat_continuation_uses_raw_units_and_owner_scale():
    # Different owners make a row/owner remapping mistake observable.
    batch = SimpleNamespace(rows=3, combat_exit_rows={1, 2},
        combat_exits=lambda **kw: {
            1: (CombatExit(.102, 'combat', value_scale=1.37), {'phase': np.array([1]), 'relics': np.zeros(1)}),
            2: (CombatExit(1.002, 'terminal', value_scale=1.002), None)},
        observations=lambda rows: {'phase': np.ones((len(rows), 1)), 'relics': np.zeros((len(rows), 1))},
        value_scales=lambda: np.array([1.402, 1.37, 1.002]))
    model = SimpleNamespace(policy=SimpleNamespace(core=SimpleNamespace(training_objective=MIXED_RAW_OBJECTIVE)),
                            action_space=SimpleNamespace(n=2), device='cpu')
    def lower(obs, width):
        return np.zeros((len(obs['phase']), 2)), np.full(len(obs['phase']), .634)
    _, values = forward_outer_leaves(model, batch, leaf_backend=lower)
    assert values == pytest.approx([.634/1.402, (.102+.634)/1.37, 1.])
