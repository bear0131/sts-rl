"""Direct upper inputs must preserve the canonical teacher's exact row bytes."""
import copy

import numpy as np
import pytest
import torch
import slaythespire as sts

from env.core import encode
from env.core.game_env import SimGameEnv
from env.core.native_observation import upper_reward_encoder
from env.core.outer_return import NORMALIZED_OBJECTIVE
from env.core.run_env import SlayTheSpireEnv
from rl.policies.frozen_upper import FrozenUpperTeacher

pytestmark = [pytest.mark.native, pytest.mark.timeout(60)]


def assert_projection(actual, expected):
    keys = (encode.OVERWORLD_OBSERVATION_KEYS |
            encode.OVERWORLD_PHASE_OBSERVATION_KEYS[encode.V.PHASE_VOCAB_INDEX['combat_reward']])
    assert actual.keys() == keys.intersection(expected)
    for key in actual:
        assert actual[key].shape == expected[key].shape, key
        assert actual[key].dtype == expected[key].dtype, key
        assert actual[key].tobytes() == expected[key].tobytes(), key


@pytest.mark.parametrize('kind', ['victory', 'stolen_victory', 'escaped_thief', 'smoke', 'loss'])
@pytest.mark.parametrize('deck_size', [False, True])
def test_search_settlement_projection_and_independent_storage(run_combat_factory, monkeypatch, kind, deck_size):
    assert hasattr(sts, 'UpperRewardObservationEncoder'), 'rebuild the explicitly selected native extension'
    root = run_combat_factory(potions=['Smoke Bomb', 'Fire Potion'])
    leaf = root.bc.clone()
    if kind in ('stolen_victory', 'escaped_thief'):
        leaf.set_player_block(999)
        leaf.execute(sts.Action(sts.ActionType.END_TURN))
    if kind in ('victory', 'stolen_victory'):
        leaf.set_player_strength(999)
        leaf.execute(sts.Action(sts.ActionType.CARD, 0, 0))
    elif kind == 'smoke':
        leaf.execute(sts.Action(sts.ActionType.POTION, 0, 0))
    elif kind == 'loss':
        leaf.apply_custom_state({'player': {'hp': 1}})
        leaf.execute(sts.Action(sts.ActionType.END_TURN))
    else:
        for _ in range(10):
            if leaf.is_battle_over:
                break
            leaf.set_player_block(999)
            leaf.execute(sts.Action(sts.ActionType.END_TURN))
    assert leaf.is_battle_over
    env = SlayTheSpireEnv(global_map_view=True, card_language=True,
                         deck_size_observation=deck_size, outer_return_context=True,
                         combat_action_contract='autoregressive-v2',
                         combat_value_objective=NORMALIZED_OBJECTIVE)
    try:
        env.reset(seed=42101)
        env._nosl_outer_branch = root.fork_run()
        before = root.obs_canonical(), leaf.to_dict()
        expected_outcome, expected = env.settle_search_combat_exit(leaf)
        # The optimized reward/terminal route must not allocate canonical obs.
        def forbidden(*args, **kwargs):
            raise AssertionError('full canonical encoder called for reward exit')
        monkeypatch.setattr(encode, 'encode', forbidden)
        actual_outcome, actual = env.settle_search_combat_exit(leaf, upper_value_only=True)
        assert actual_outcome == expected_outcome
        assert before == (root.obs_canonical(), leaf.to_dict())
        if kind == 'loss':
            assert actual is expected is None
            return
        assert_projection(actual, expected)
        saved = {key: value.copy() for key, value in actual.items()}
        _, another = env.settle_search_combat_exit(leaf, upper_value_only=True)
        for key in another:
            another[key].fill(99)
            assert actual[key].tobytes() == saved[key].tobytes()
    finally:
        env.close()


@pytest.mark.parametrize('seed', [21, 42101, 9030000])
def test_real_run_reward_screens_across_acts(seed):
    root = SimGameEnv(global_map_view=True)
    root.reset(seed=seed)
    root.gc.skip_battles = True
    seen = set()
    screens = 0
    codec = upper_reward_encoder(True)
    for _ in range(600):
        if root.is_over:
            break
        if root.phase() == 'combat_reward':
            expected = encode.encode(root.encoding_canonical(), global_map_view=True,
                                     card_language=True, deck_size_observation=True)
            assert_projection(codec.encode(root.gc), expected)
            seen.add(root.gc.act)
            screens += 1
        actions = root.legal_semantic_actions()
        assert actions
        # Claim rewards (including card packs), then take the first legal route.
        root.step(actions[0])
    assert seen == {1, 2, 3} and screens >= 20


def test_teacher_projection_forward_and_screen_rejection(frozen_upper_source, run_combat_factory):
    core, space, env, _ = frozen_upper_source
    teacher = FrozenUpperTeacher(core, space, source_sha256='1'*64)
    root = run_combat_factory()
    codec = upper_reward_encoder(False)
    with pytest.raises(ValueError, match='settled reward screen'):
        codec.encode(root.gc)
    leaf = root.bc.clone()
    leaf.set_player_strength(999)
    leaf.execute(sts.Action(sts.ActionType.CARD, 0, 0))
    branch, _ = root.settle_combat_leaf(leaf)
    clone = copy.copy(env)
    clone.thick = branch
    clone._invalidate_backend_cache()
    full = clone._obs()
    rows = [codec.encode(branch.gc) for _ in range(3)]
    # Change actual live state between calls: there is no per-root observation cache.
    branch.gc.gold += 17
    branch.gc.set_deck([('SEARING_BLOW', 5), ('STRIKE_RED', 0), ('STRIKE_RED', 1)])
    branch.gc.set_relic_state([dict(id='Pen Nib', counter=7), dict(id='Ancient Tea Set', counter=1)])
    branch.gc.set_potion_state(['Fire Potion', 'Smoke Bomb'])
    branch.gc.cur_hp = 47
    branch.gc.max_hp = 137
    changed = clone._obs()
    rows.append(codec.encode(branch.gc))
    assert_projection(rows[-1], changed)
    expected = teacher.prepare_search_numpy([full, full, full, changed])
    actual = teacher.prepare_search_numpy(rows)
    for (ei, ep, ew, eb), (ai, ap, aw, ab) in zip(expected, actual, strict=True):
        assert ep == ap and ew == aw and eb.keys() == ab.keys()
        np.testing.assert_array_equal(ei, ai)
        for key in eb:
            assert eb[key].tobytes() == ab[key].tobytes(), key
        with torch.no_grad():
            e = teacher.forward_phase({k: torch.as_tensor(v) for k, v in eb.items()}, ep, ew)
            a = teacher.forward_phase({k: torch.as_tensor(v) for k, v in ab.items()}, ap, aw)
        for x, y in zip(e, a, strict=True):
            torch.testing.assert_close(x, y, rtol=0, atol=0)


def test_multiple_reward_packs_and_current_pack_after_claim(run_combat_factory):
    root = run_combat_factory()
    root.gc.set_relics(['Prayer Wheel', 'Question Card'])
    root.bc = sts.create_battle(root.gc, 'Looter')
    root.bc.set_player_strength(999)
    root.bc.execute(sts.Action(sts.ActionType.CARD, 0, 0))
    root._settle()
    assert root.gc.get_reward_info()['card_count'] == 2
    codec = upper_reward_encoder(False)
    for count in (2, 1, 0):
        assert root.gc.get_reward_info()['card_count'] == count
        assert_projection(codec.encode(root.gc), encode.encode(root.encoding_canonical()))
        if count:
            root.gc.skip_reward_cards()


def test_native_reward_transaction_preserves_source_and_avoids_python_settlement(run_combat_factory, monkeypatch):
    assert hasattr(sts.UpperRewardObservationEncoder, 'settle_reward'), 'rebuild the selected native extension'
    root = run_combat_factory()
    leaf = root.bc.clone()
    leaf.set_player_strength(999)
    leaf.execute(sts.Action(sts.ActionType.CARD, 0, 0))
    env = SlayTheSpireEnv(outer_return_context=True, combat_action_contract='autoregressive-v2',
                         combat_value_objective=NORMALIZED_OBJECTIVE)
    try:
        env.reset(seed=42101)
        env._nosl_outer_branch = root.fork_run()
        source_before = root.obs_canonical(), leaf.to_dict()
        expected_exit, expected = env.settle_search_combat_exit(leaf)
        def forbidden(*args, **kwargs):
            raise AssertionError('reward transaction returned through Python settlement')
        monkeypatch.setattr(SimGameEnv, '_settle_combat_leaf', forbidden)
        actual_exit, actual = env.settle_search_combat_exit(leaf, upper_value_only=True)
        assert actual_exit == expected_exit
        assert_projection(actual, expected)
        assert (root.obs_canonical(), leaf.to_dict()) == source_before
    finally:
        env.close()
