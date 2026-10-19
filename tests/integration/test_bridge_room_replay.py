"""BRIDGE39 (main BRIDGE29): replay room exits through native continuations."""
import pytest
import slaythespire as sts

from bridge.replay import Driver
from env.core.run_env import SlayTheSpireEnv

pytestmark = pytest.mark.native


@pytest.fixture(params=[0, 20])
def room_driver(request):
    created = []

    def make(room, relics=()):
        env = SlayTheSpireEnv(ascension=request.param, combat_action_contract='native-v1')
        env.reset(seed=42)
        created.append(env)
        gc = env.thick.gc
        x, y = next((x, y) for y in range(15) for x in range(7)
                    if env.thick.rmap.room_type(x, y) == room)
        gc.set_relics(list(relics))
        gc.cur_map_node_y = y - 1
        gc.cur_map_node_x = next(parent for parent in range(7)
                                if x in env.thick.rmap.reachable(parent, y - 1))
        gc.floor_num = y
        gc.screen_state = sts.ScreenState.MAP_SCREEN
        gc.transition_to_map_node(x)
        return Driver(gc)

    yield make
    for env in created:
        env.close()


def resources(gc):
    return (gc.gold, gc.cur_hp, gc.max_hp, gc.deck_ids(), gc.relic_state(),
            gc.floor_num, gc.red_key, gc.blue_key, gc.green_key)


def test_replay_campfire_without_available_actions_can_leave(room_driver):
    driver = room_driver('REST', ['Coffee Dripper', 'Fusion Hammer'])
    gc = driver.gc
    gc.red_key = True
    assert gc.legal_campfire_options() == [6]
    before = resources(gc)
    driver.overworld_cmd({'screen_type': 'REST'}, 'proceed')
    assert gc.screen_state == sts.ScreenState.MAP_SCREEN
    assert resources(gc) == before


@pytest.mark.parametrize('option', ['rest', 'smith'])
def test_replay_campfire_followup_does_not_repeat_completed_action(room_driver, option):
    driver = room_driver('REST')
    gc = driver.gc
    gc.cur_hp = 20
    driver.overworld_cmd({'screen_type': 'REST', 'choice_list': [option]}, 'choose 0')
    if option == 'smith':
        assert gc.screen_state == sts.ScreenState.CARD_SELECT
        gc.choose_select_card_option(0)
        assert any(up for _, up in gc.deck_ids())
    else:
        assert gc.cur_hp > 20
    assert gc.screen_state == sts.ScreenState.MAP_SCREEN
    before = resources(gc)
    driver.overworld_cmd({'screen_type': 'REST'}, 'proceed')
    assert gc.screen_state == sts.ScreenState.MAP_SCREEN
    assert resources(gc) == before


@pytest.mark.parametrize('exit_screen', ['REST', 'COMBAT_REWARD'])
def test_replay_campfire_reward_can_be_left(room_driver, exit_screen):
    driver = room_driver('REST', ['Dream Catcher'])
    gc = driver.gc
    driver.overworld_cmd({'screen_type': 'REST', 'choice_list': ['rest']}, 'choose 0')
    assert gc.screen_state == sts.ScreenState.REWARDS
    assert gc.get_reward_info()['card_count'] == 1
    before = resources(gc)
    driver.overworld_cmd({'screen_type': exit_screen}, 'proceed')
    assert gc.screen_state == sts.ScreenState.MAP_SCREEN
    assert resources(gc) == before
    driver.overworld_cmd({'screen_type': 'REST'}, 'proceed')
    assert resources(gc) == before


@pytest.mark.parametrize('opened', [False, True])
def test_replay_chest_proceed_skips_only_current_room_rewards(room_driver, opened):
    driver = room_driver('TREASURE', ['Cursed Key'])
    gc = driver.gc
    initial_cards = len(gc.deck_ids())
    if opened:
        driver.overworld_cmd({'screen_type': 'CHEST'}, 'choose 0')
        assert gc.screen_state == sts.ScreenState.REWARDS
        assert len(gc.deck_ids()) == initial_cards + 1
    else:
        assert gc.screen_state == sts.ScreenState.TREASURE_ROOM
    before = resources(gc)
    driver.overworld_cmd({'screen_type': 'CHEST'}, 'proceed')
    assert gc.screen_state == sts.ScreenState.MAP_SCREEN
    assert resources(gc) == before
    # The real UI can still show the room after the sim consumed its continuation.
    driver.overworld_cmd({'screen_type': 'CHEST'}, 'proceed')
    assert resources(gc) == before


def test_replay_chest_after_reward_proceed_is_already_complete(room_driver):
    driver = room_driver('TREASURE')
    gc = driver.gc
    driver.overworld_cmd({'screen_type': 'CHEST'}, 'choose 0')
    info = gc.get_reward_info()
    assert info['relics']
    driver.overworld_cmd({'screen_type': 'COMBAT_REWARD', 'choice_list': ['relic']}, 'choose 0')
    assert info['relics'][0] in gc.relic_ids()
    driver.overworld_cmd({'screen_type': 'COMBAT_REWARD'}, 'proceed')
    before = resources(gc)
    driver.overworld_cmd({'screen_type': 'CHEST'}, 'proceed')
    assert gc.screen_state == sts.ScreenState.MAP_SCREEN
    assert resources(gc) == before


def test_replay_cannot_skip_an_available_campfire_choice(room_driver):
    driver = room_driver('REST')
    gc = driver.gc
    assert 6 not in gc.legal_campfire_options()
    before = resources(gc)
    with pytest.raises(ValueError, match='campfire actions'):
        driver.overworld_cmd({'screen_type': 'REST'}, 'proceed')
    assert gc.screen_state == sts.ScreenState.REST_ROOM
    assert resources(gc) == before


@pytest.mark.parametrize('real_screen', ['REST', 'CHEST'])
def test_replay_proceed_cannot_discard_pending_selection(room_driver, real_screen):
    driver = room_driver('REST')
    gc = driver.gc
    driver.overworld_cmd({'screen_type': 'REST', 'choice_list': ['smith']}, 'choose 0')
    assert gc.screen_state == sts.ScreenState.CARD_SELECT
    before = resources(gc), gc.get_select_cards()
    with pytest.raises(ValueError, match='pending sim screen'):
        driver.overworld_cmd({'screen_type': real_screen}, 'proceed')
    assert gc.screen_state == sts.ScreenState.CARD_SELECT
    assert (resources(gc), gc.get_select_cards()) == before
