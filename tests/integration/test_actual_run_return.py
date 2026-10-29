"""Raw full-run labels and search-only scaling, without upper critic calls."""
from types import SimpleNamespace
import copy
import numpy as np
import pytest

from env.core import encode, vocab as V, native_actions as C
from env.core.outer_return import MC_OBJECTIVE, MC_EXIT_KEY, remaining_reward_max
from env.core.run_env import SlayTheSpireEnv
from env.search.native_batch import NativeSearchBatch
from rl.algorithms import outer_return as outer
from rl.search.native import forward_outer_leaves

pytestmark = [pytest.mark.native, pytest.mark.timeout(30)]


def frame(act, floor, phase='combat'):
    return dict(act=np.array([[act / encode.ACT_SCALE]], dtype=np.float32),
                floor=np.array([[floor / encode.FLOOR_SCALE]], dtype=np.float32),
                phase=np.array([[V.PHASE_VOCAB_INDEX[phase]]]))


def test_labels_wait_for_run_end_and_include_upper_rewards_without_normalizing():
    teacher = SimpleNamespace(assert_integrity=lambda: None)
    model = SimpleNamespace(policy=SimpleNamespace(core=SimpleNamespace(
        training_objective=MC_OBJECTIVE, teacher=teacher)), action_space=SimpleNamespace(n=198),
        _nosl_is_az_pending_fights=[[]], _nosl_is_az_samples=[])
    outer.begin_collection(model, 1)
    pending = model._nosl_is_az_pending_fights
    model._last_obs = frame(1, 0)
    pending[0].append({})
    outer.stamp_roots(model, pending)
    exit_obs = {MC_EXIT_KEY: np.ones(1, dtype=np.float32)}
    info = dict(run_stats=dict(act=1, floor=1, outcome='UNDECIDED'),
                nosl_value_terminal_observation=exit_obs)
    outer.finish_step(model, frame(1, 0), frame(1, 1, 'map'), [False], [info], pending)
    assert len(pending[0]) == 2 and not model._nosl_is_az_samples
    assert pending[0][-1]['policy_weight'] == 0
    assert all(s['outer_reward_prefix'] == 0 for s in pending[0])
    # A collection boundary preserves both reward prefix and unlabeled states.
    restored = copy.deepcopy(model)
    outer.begin_collection(restored, 1)
    for m in (model, restored):
        p = m._nosl_is_az_pending_fights
        outer.finish_step(m, frame(1, 1, 'map'), frame(2, 17, 'map'), [False],
                          [dict(run_stats=dict(act=2, floor=17, outcome='UNDECIDED'))], p)
        assert not m._nosl_is_az_samples
        m._last_obs = frame(2, 17)
        p[0].append({})
        outer.stamp_roots(m, p)
        outer.finish_step(m, frame(2, 17), frame(1, 0), [True],
                          [dict(run_stats=dict(act=4, floor=51, outcome='PLAYER_VICTORY', seed=7))], p)
        targets = [s['value_target'] for s in m._nosl_is_az_samples]
        assert targets == pytest.approx([1.402, 1.402, 1.268])
        assert not p[0] and m._outer_pending_rewards[0] == 0
        assert m._outer_collection['collector_upper_batches'] == 0


def test_censored_run_drops_all_pending_without_zero_or_bootstrap_labels():
    model = SimpleNamespace(policy=SimpleNamespace(core=SimpleNamespace(
        training_objective=MC_OBJECTIVE, teacher=SimpleNamespace(assert_integrity=lambda: None))),
        action_space=SimpleNamespace(n=198), _nosl_is_az_pending_fights=[[]], _nosl_is_az_samples=[])
    outer.begin_collection(model, 1)
    model._nosl_is_az_pending_fights[0].append(dict(outer_reward_prefix=0.))
    result = outer.finish_step(model, frame(1, 0), frame(1, 0), [True],
        [{'TimeLimit.truncated': True}], model._nosl_is_az_pending_fights)
    assert result == (0, 1, 0) and not model._nosl_is_az_samples


@pytest.mark.parametrize('direct', [False, True])
@pytest.mark.parametrize('kind', ['victory', 'escape', 'loss'])
def test_terminal_search_uses_lower_raw_value_once_scaled_and_marks_exit(run_combat_factory, direct, kind):
    backend = run_combat_factory(potions=['Smoke Bomb'] if kind == 'escape' else [])
    if kind == 'victory':
        backend.bc.set_player_strength(999)
        action = C.AUTOREGRESSIVE_PLAY
    elif kind == 'escape':
        action = C.AUTOREGRESSIVE_POTION + 5
    else:
        backend.bc.apply_custom_state({'player': {'hp': 1}})
        action = 0
    env = SlayTheSpireEnv(outer_return_context=True, combat_action_contract='autoregressive-v2',
                         combat_value_objective=MC_OBJECTIVE)
    env.reset(seed=42101)
    backend._actual_run_return = True
    env.thick = backend
    env._invalidate_backend_cache()
    root_obs = env._obs()
    batch = NativeSearchBatch([env], 1, direct_observations=direct)
    try:
        logits = np.full(198, -1000., dtype=np.float32)
        logits[action] = 1000.
        batch.begin({0: ((logits,), dict(simulations=1, depth=0, root_value=1.402,
            state_nonce=42101, fixed_worldline=True, outer_return=True, compact_observation=True))})
        assert list(batch.combat_exit_rows) == [0]
        before = backend.bc.to_dict()
        def predict(obs, width):
            assert obs[MC_EXIT_KEY].tolist() == [[1.]]
            return np.zeros((1, 198), dtype=np.float32), np.array([1.402], dtype=np.float32)
        model = SimpleNamespace(policy=SimpleNamespace(core=SimpleNamespace(training_objective=MC_OBJECTIVE)))
        def forbidden(**kwargs):
            raise AssertionError('Monte Carlo search must not settle or query upper values')
        batch.combat_exits = forbidden
        _, values = forward_outer_leaves(model, batch, leaf_backend=predict)
        assert values[0] == pytest.approx(1.402 / remaining_reward_max(backend.gc.act, backend.gc.floor_num))
        assert backend.bc.to_dict() == before
        # The actual transition exposes the same pre-settlement terminal inputs.
        _, _, _, _, info = env.step(action)
        terminal = info['nosl_value_terminal_observation']
        assert terminal[MC_EXIT_KEY].tolist() == [1.]
        for key in ('act', 'floor', 'deck_ids', 'deck_upgrades'):
            np.testing.assert_array_equal(terminal[key], root_obs[key])
    finally:
        batch.close()
        env.close()
