"""Act 4 entry, reward exclusivity, and the mechanics used by Heart runs."""
import json
from pathlib import Path

import pytest
import slaythespire as sts

from env.core.game_env import SimGameEnv
from env.core.run_env import SlayTheSpireEnv

pytestmark = pytest.mark.native


def _env():
    env = SlayTheSpireEnv(
        backend_factory=lambda: SimGameEnv(ascension=20, stop_after_act=4),
        combat_action_contract='native-v1')
    env.reset(seed=4214060967)
    return env


def _original_actions():
    return json.loads((Path(__file__).parents[1] / 'fixtures' /
                       'a20_infinite_seed4214060967.json').read_text())['actions']


@pytest.mark.parametrize('take_key', [True, False])
def test_chest_key_and_relic_are_mutually_exclusive(take_key):
    env = _env()
    try:
        for kind, action in _original_actions():
            if env.thick.gc.floor_num == 43 and kind == 'thin' and action == 126:
                break
            (env.step_native_combat if kind == 'native' else env.step)(action)
        before = env.thick.gc.get_reward_info()
        assert before['sapphire_key'] and len(before['relics']) == 1
        if take_key:
            env.step_key_reward('sapphire')
            assert env.thick.gc.blue_key
            assert env.thick.gc.get_reward_info()['relics'] == []
        else:
            env.step(126)
            assert not env.thick.gc.blue_key
        assert env.thick.legal_key_rewards() == []
        with pytest.raises(ValueError, match='unavailable'):
            env.step_key_reward('sapphire')
        with pytest.raises(ValueError, match='no claimable'):
            env.thick.gc.claim_key_reward()
    finally:
        env.close()


def test_three_keys_continue_after_second_a20_boss():
    env = _env()
    try:
        for kind, action in _original_actions():
            if 'emerald' in env.thick.legal_key_rewards():
                relics = env.thick.gc.get_reward_info()['relics']
                env.step_key_reward('emerald')
                assert env.thick.gc.get_reward_info()['relics'] == relics
            floor = env.thick.gc.floor_num
            if floor == 43 and kind == 'thin' and action == 126:
                env.step_key_reward('sapphire')
                continue
            if floor == 49 and kind == 'thin' and action == 141:
                env.step(142)  # recall instead of smith
                continue
            if floor == 49 and kind == 'thin' and action == 22:
                continue  # old smith selection
            result = (env.step_native_combat if kind == 'native' else env.step)(action)
            assert not result[2] and not result[3]
        assert env.thick.gc.act == 4
        assert env.thick.gc.blue_key and env.thick.gc.green_key and env.thick.gc.red_key
        assert env.thick.gc.outcome == sts.GameOutcome.UNDECIDED
        assert env.thick.rmap.reachable(-1, -1) == [3]
        assert [env.thick.rmap.room_type(3, y) for y in range(4)] == [
            'REST', 'SHOP', 'ELITE', 'BOSS']
        # Returning to this live point through the native-key journal is exact.
        state = env.export_training_state()
        clone = _env()
        try:
            clone.restore_training_state(state)
            assert clone.thick.obs_canonical() == env.thick.obs_canonical()
        finally:
            clone.close()
        env.step(120)
        assert env.thick.phase() == 'rest' and not env.thick.is_over
    finally:
        env.close()


def _battle(encounter, deck):
    gc = sts.GameContext(sts.CharacterClass.IRONCLAD, 42, 20)
    gc.floor_num = 55
    gc.max_hp = 999
    gc.cur_hp = 999
    gc.set_deck([(card, 0) for card in deck])
    gc.set_relics([])
    return gc, sts.create_battle(gc, encounter)


def test_spear_a20_puts_two_burns_on_top():
    gc, bc = _battle('Shield And Spear', ['STRIKE_RED'] * 15)
    bc.execute(sts.Action(4 << 29))
    hand = bc.to_dict()['hand']
    assert [c[0] for c in hand[:2]] == ['BURN', 'BURN']
    assert sum(c[0] == 'BURN' for c in hand) == 2


def test_heart_a20_damage_cap_and_beat_of_death():
    gc, bc = _battle('The Heart', ['STRIKE_RED'] * 15)
    bc.set_player_strength(999)
    before = bc.to_dict()
    bc.execute(sts.Action(0))
    first = bc.to_dict()
    bc.execute(sts.Action(0))
    second = bc.to_dict()
    assert first['monsters'][0]['hp'] == second['monsters'][0]['hp'] == 600
    assert before['player']['hp'] - first['player']['hp'] == 2
    assert first['player']['hp'] - second['player']['hp'] == 2
    bc.execute(sts.Action(4 << 29))
    bc.execute(sts.Action(0))
    assert bc.to_dict()['monsters'][0]['hp'] == 400
