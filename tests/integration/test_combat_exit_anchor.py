from copy import deepcopy

import numpy as np
import pytest
import slaythespire as sts

from env.core import native_actions as C
from env.core.run_env import SlayTheSpireEnv
from env.core.combat_value import combat_terminal_outcome
from schemas.combat_state import normalize_sim_global

pytestmark = [pytest.mark.native, pytest.mark.timeout(90)]


def make_env():
    return SlayTheSpireEnv(global_map_view=True, upper_public_context=True, event_option_observation=True,
        card_language=True, combat_action_contract='autoregressive-v2',
        combat_exit_anchor=True, reward_flow='sequential-v1')


@pytest.mark.parametrize('outcome', ['win', 'escape', 'loss'])
def test_real_terminal_boundary_precedes_rewards_and_preserves_settlement(run_combat_factory, outcome):
    env = make_env()
    env.reset(seed=42101)
    root = run_combat_factory(potions=['Smoke Bomb'], global_map_view=True, upper_public_context=True)
    root.combat_exit_anchor_enabled = True
    env.thick = root
    env._invalidate_backend_cache()
    if outcome == 'win':
        root.bc.set_player_strength(999)
        action = next(a for a in np.flatnonzero(env.action_masks()) if 1 <= a < 61)
    elif outcome == 'escape':
        action = next(a for a in np.flatnonzero(env.action_masks()) if 61 <= a < 91)
    else:
        root.bc.apply_custom_state({'player': {'hp': 1}})
        action = C.AUTOREGRESSIVE_END
    before = normalize_sim_global(root.gc, combat=True)
    expected = root.fork_run()
    observation, reward, done, truncated, info = env.step(int(action))
    assert not truncated
    if outcome == 'loss':
        assert done and root.bc is None
        assert not observation['combat_exit_anchor'].any()
        return
    assert not done and root.bc.is_battle_over
    assert combat_terminal_outcome(root.bc) == outcome
    assert normalize_sim_global(root.gc, combat=True) == before
    assert root.gc.screen_state_name.endswith('BATTLE')
    np.testing.assert_array_equal(observation['combat_exit_anchor'],
                                  [outcome == 'win', outcome == 'escape'])
    assert np.flatnonzero(env.action_masks()).tolist() == [C.AUTOREGRESSIVE_END]
    assert '结算' in env.describe_action(C.AUTOREGRESSIVE_END)
    assert '尚未' in env.render()
    for key in ('offer_card_ids', 'reward_gold', 'reward_relic_ids', 'reward_potion_ids', 'reward_keys'):
        assert not observation[key].any(), key
    expected.bc = root.bc.clone()
    expected._settle(release_anchor=True)
    battle = root.bc.to_dict()
    # Inspecting the anchor repeatedly is pure and does not consume its RNG.
    for _ in range(3):
        env._obs(); env.action_masks()
    assert root.bc.to_dict() == battle and normalize_sim_global(root.gc, combat=True) == before
    after, _, done, truncated, info = env.step(C.AUTOREGRESSIVE_END)
    assert root.obs_canonical() == expected.obs_canonical()
    assert root.gc.clone()._dbg_card_reward() == expected.gc.clone()._dbg_card_reward()
    assert not after['combat_exit_anchor'].any() and not truncated
    assert info['combat_anchor_continue']


def test_natural_anchor_and_sequential_stage_survive_action_journal_replay():
    env, restored = make_env(), make_env()
    try:
        env.reset(seed=42101)
        anchor = False
        for _ in range(500):
            if env._at_combat_exit_anchor():
                anchor = True
                break
            legal = np.flatnonzero(env.action_masks())
            # Play available cards/potions before ending a turn; outside combat
            # the first offered choice suffices for this bounded real episode.
            cards = legal[(legal > 0) & (legal < 91)] if env._phase() == 'combat' else []
            result = env.step(int(cards[0] if len(cards) else legal[0]))
            assert not result[2] and not result[3], 'fixture must reach a natural successful fight'
        assert anchor
        state = env.export_training_state()
        restored.restore_training_state(state)
        for _ in range(8):
            a, b = env._obs(), restored._obs()
            for key in a:
                np.testing.assert_array_equal(a[key], b[key], err_msg=key)
            np.testing.assert_array_equal(env.action_masks(), restored.action_masks())
            action = int(np.flatnonzero(env.action_masks())[-1])
            left, right = env.step(action), restored.step(action)
            assert left[1:4] == right[1:4]
        with pytest.raises(ValueError, match='anchor contract'):
            SlayTheSpireEnv(combat_action_contract='autoregressive-v2').restore_training_state(state)
    finally:
        env.close(); restored.close()
