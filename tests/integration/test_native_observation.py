"""Native execution can supply checkpoint observations without a second env."""
import copy

import numpy as np
import pytest

from env.core import encode, native_actions as C  # native bootstrap before sts
from env.core.native_observation import encode_native_observation
import slaythespire as sts
from schemas.combat_state import normalize_sim_global


def _assert_bytes(actual, expected):
    assert actual.keys() <= expected.keys()
    for key, value in actual.items():
        assert value.shape == expected[key].shape, key
        assert value.dtype == expected[key].dtype, key
        assert value.tobytes() == expected[key].tobytes(), key


@pytest.mark.parametrize('combat_only', [False, True])
def test_native_branch_ar_prefix_matches_live_ar_environment(combat_factory, combat_only):
    native, reference = [combat_factory(['PURITY'] * 5,
        combat_action_contract=contract, card_language=True)
        for contract in ('native-v1', C.AUTOREGRESSIVE_CONTRACT)]
    for env in (native, reference):
        env.thick.bc.apply_custom_state({
            'hand': ['PURITY', 'STRIKE_RED', 'DEFEND_RED', 'BASH', 'ANGER']})
        env.thick.bc.execute(sts.Action(sts.ActionType.CARD, 0))
        env._invalidate_backend_cache()
    raw = native.thick.bc.to_dict()
    global_state = normalize_sim_global(native.thick.gc)
    original_raw, original_global = copy.deepcopy(raw), copy.deepcopy(global_state)
    for prefix, picked_slot in ((0, None), (1, 0), (5, 2)):
        if picked_slot is not None:
            reference.step(C.AUTOREGRESSIVE_MULTI + picked_slot)
        state = dict(raw, selection_prefix=prefix)
        policy_ids = [aid for aid, _ in native.thick.bc.autoregressive_policy_actions(prefix)]
        actual = encode_native_observation(state, root_global=global_state,
            combat_action_contract=C.AUTOREGRESSIVE_CONTRACT, policy_ids=policy_ids,
            card_language=True, combat_only=combat_only)
        _assert_bytes(actual, reference._obs())
        np.testing.assert_array_equal(actual['combat_legal'], reference.action_masks())
        assert np.flatnonzero(actual['select_hand_mask']).tolist() == [i for i in range(10) if prefix >> i & 1]
        assert actual['pick_remaining'][0] == pytest.approx((3 - prefix.bit_count()) / encode.PICK_SCALE)
    assert raw == original_raw and global_state == original_global
    assert native.thick.bc.to_dict() == original_raw
    assert native.combat_action_contract == 'native-v1'


@pytest.mark.parametrize('card,zone', [
    ('SECRET_TECHNIQUE', 'draw'), ('HEADBUTT', 'discard'), ('EXHUME', 'exhaust')])
def test_branch_selection_candidates_keep_native_pile_handles(combat_factory, card, zone):
    env = combat_factory([card] * 5, combat_action_contract='native-v1')
    env.thick.bc.apply_custom_state({'hand': [card],
        zone: ['STRIKE_RED'] * 12 + ['SHRUG_IT_OFF', 'DEFEND_RED']})
    env.thick.bc.execute(sts.Action(sts.ActionType.CARD, 0, 0))
    raw = env.thick.bc.to_dict()
    candidates = raw['select']['candidates']
    assert any(c['native_index'] >= 12 for c in candidates)
    policy_ids = [aid for aid, _ in env.thick.bc.autoregressive_policy_actions(0)]
    obs = encode_native_observation(raw, root_global=normalize_sim_global(env.thick.gc),
        combat_action_contract=C.AUTOREGRESSIVE_CONTRACT, policy_ids=policy_ids, combat_only=True)
    np.testing.assert_array_equal(obs['select_native_indices'][:len(candidates)],
                                  [c['native_index'] + 1 for c in candidates])
    # The public candidate slot resolves the same entity as the native handle.
    for index, candidate in enumerate(candidates):
        assert obs['combat_legal'][C.AUTOREGRESSIVE_PICK + index]
        branch = env.thick.bc.clone()
        branch.execute(sts.Action(sts.ActionType.SINGLE_CARD_SELECT, candidate['native_index']))
        assert branch.to_dict()['input_state'] != 2


@pytest.mark.parametrize('combat_only', [False, True])
def test_branch_hp_potions_dead_slots_and_root_metadata_are_independent(combat_factory, combat_only):
    env = combat_factory(['STRIKE_RED'] * 5, 'Three Sentries',
        combat_action_contract='native-v1', relics=['Anchor'])
    env.thick.bc.apply_custom_state({'player': {'hp': 40, 'max_hp': 80, 'gold': 77},
        'potions': ['Blood Potion'], 'monsters': [{'hp': 0}, {'hp': 40}, {'hp': 0}]})
    global_state = normalize_sim_global(env.thick.gc)
    frozen_global = copy.deepcopy(global_state)
    root = env.thick.bc.to_dict()
    branch = env.thick.bc.clone()
    branch.execute(sts.Action(sts.ActionType.POTION, 0))
    raw = branch.to_dict()
    assert raw['player']['hp'] > root['player']['hp']
    before, after = [encode_native_observation(state, root_global=global_state,
        combat_action_contract='native-v1', combat_only=combat_only) for state in (root, raw)]
    assert after['player_scalars'][0] == pytest.approx(raw['player']['hp'] / encode.HP_SCALE)
    if not combat_only:
        assert after['run_hp'][0] == after['player_scalars'][0]
    assert before['potions'][0] != 0 and after['potions'][0] == 0
    assert after['gold'][0] == pytest.approx(77 / encode.GOLD_SCALE)
    assert after['relics'].any()
    np.testing.assert_array_equal(after['relics'], before['relics'])
    assert (after['monster_id'][:3] != 0).all()
    np.testing.assert_array_equal(after['monster_scalars'][:3, 0], [0, 1, 0])
    assert 'combat_legal' not in after
    assert global_state == frozen_global and env.thick.bc.to_dict() == root


@pytest.mark.parametrize('language', [False, True])
def test_hidden_draw_order_keeps_counts_and_language_group_multiset(combat_factory, language):
    env = combat_factory(['STRIKE_RED'] * 5)
    env.thick.bc.apply_custom_state({'draw': [
        ('STRIKE_RED', 0), ('DEFEND_RED', 1), ('STRIKE_RED', 2), ('BASH', 0)]})
    raw = env.thick.bc.to_dict()
    shuffled = copy.deepcopy(raw)
    shuffled['draw'].reverse()
    observations = [encode_native_observation(state,
        root_global=normalize_sim_global(env.thick.gc), combat_action_contract='native-v1',
        card_language=language, combat_only=True) for state in (raw, shuffled)]
    if not language:
        _assert_bytes(*observations)
    else:
        # Existing language encoding groups in first-occurrence order. Preserve
        # that checkpoint contract; the multiset, not its row order, is invariant.
        keys = ('draw_card_ids', 'draw_card_counts', 'draw_card_upgraded_counts', 'draw_card_upgrades')
        groups = [sorted(zip(*(o[key].tolist() for key in keys))) for o in observations]
        assert groups[0] == groups[1]
        _assert_bytes({k: v for k, v in observations[0].items() if k not in keys},
                      {k: v for k, v in observations[1].items() if k not in keys})


@pytest.mark.parametrize('flag', [
    'global_map_view', 'card_language', 'combat_language', 'deck_size_observation',
    'neow_observation', 'event_option_observation', 'interactive_reasoning', 'all'])
def test_explicit_observation_flags_keep_declared_checkpoint_schema(combat_factory, flag):
    env = combat_factory(['STRIKE_RED'] * 5, combat_action_contract='native-v1')
    flags = {name: flag in (name, 'all') for name in (
        'global_map_view', 'card_language', 'combat_language', 'deck_size_observation',
        'neow_observation', 'event_option_observation', 'interactive_reasoning')}
    policy_ids = [aid for aid, _ in env.thick.bc.autoregressive_policy_actions(0)]
    obs = encode_native_observation(env.thick.bc.to_dict(),
        root_global=normalize_sim_global(env.thick.gc),
        combat_action_contract=C.AUTOREGRESSIVE_CONTRACT, policy_ids=policy_ids, **flags)
    legal = obs.pop('combat_legal')
    assert legal.shape == (C.AUTOREGRESSIVE_WIDTH,) and legal.dtype == np.bool_
    assert encode.obs_space(**flags).contains(obs)
