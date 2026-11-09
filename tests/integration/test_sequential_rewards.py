import numpy as np
import pytest

from env.core import action_space as A
from env.core.run_env import SlayTheSpireEnv
from env.core.reward_flow import STAGES

pytestmark = [pytest.mark.native, pytest.mark.timeout(90)]


@pytest.fixture
def reward_env(upper_search_env_factory):
    source = upper_search_env_factory('combat_reward', room_type='ELITE',
        relics=['Black Star', 'White Beast Statue'])
    env = SlayTheSpireEnv(global_map_view=True, upper_public_context=True, card_language=True, event_option_observation=True,
        combat_action_contract='autoregressive-v2', reward_flow='sequential-v1')
    env.reset(seed=42101)
    env.thick = source.thick.fork_run()
    env._invalidate_backend_cache()
    offers = env.thick.gc.get_reward_info()
    assert len(offers['relics']) == 2 and offers['gold'] and offers['potions'] and offers['card_count']
    yield env
    env.close()


def stage(env):
    return STAGES[int(env._obs()['reward_stage'].argmax())]


def advance_to(env, target):
    for _ in range(16):
        if stage(env) == target:
            return
        env.step(A._RSKIP if stage(env) == 'card' else A._PROCEED)
    raise AssertionError('failed to reach reward stage')


def test_relic_first_hide_future_and_preserve_native_indices(reward_env):
    env = reward_env
    assert stage(env) == 'relic'
    assert env.describe_action(A._PROCEED) == '跳过当前奖励'
    assert env.render().startswith('遗物：')
    before = env.thick.gc.get_reward_info()
    obs = env._obs()
    assert np.flatnonzero(obs['reward_relic_ids']).tolist() == [0]
    for key in ('reward_gold', 'reward_potion_ids', 'offer_card_ids'):
        assert not obs[key].any()
    assert env.action_masks()[A._RRELIC]
    env.step(A._PROCEED)  # skip only relic zero, do not leave the reward page
    assert stage(env) == 'relic'
    assert env.thick.gc.get_reward_info()['relics'] == before['relics']
    assert np.flatnonzero(env._obs()['reward_relic_ids']).tolist() == [1]
    assert not env.action_masks()[A._RRELIC]
    assert env.action_masks()[A._RRELIC + 1]
    result = env.step(A._RRELIC + 1)
    # A bottled relic is allowed to open its real selection before returning.
    for _ in range(8):
        if env._phase() != 'card_select':
            break
        env.step(int(np.flatnonzero(env.action_masks())[0]))
    assert env.thick.gc.get_reward_info()['relics'] == [before['relics'][0]]
    assert stage(env) == 'gold'
    assert result[4]['reward_flow']['taken']
    assert not env._obs()['reward_relic_ids'].any()


@pytest.mark.parametrize('blocked', [False, True])
def test_full_potion_belt_replace_or_sozu_rejection(reward_env, blocked):
    env = reward_env
    advance_to(env, 'potion')
    gc = env.thick.gc
    gc.set_potion_state(['Fire Potion'] * gc.potion_capacity)
    if blocked:
        gc.set_relics(['Sozu'])
    env._invalidate_backend_cache()
    offers = gc.get_reward_info()['potions']
    assert not env._obs()['offer_card_ids'].any()
    assert bool(env.action_masks()[A._RPOTION]) == (not blocked)
    if blocked:
        result = env.step(A._PROCEED)
        assert not result[4]['reward_flow']['available']
        assert gc.get_reward_info()['potions'] == offers
        return
    result = env.step(A._RPOTION)
    assert stage(env) == 'replace_potion'
    assert not result[4]['reward_flow']['taken']
    before = list(gc.overworld_potions)
    assert env.action_masks()[A._POTION + 1]
    result = env.step(A._POTION + 1)
    assert result[4]['reward_flow']['taken']
    assert gc.get_reward_info()['potions'] == offers[1:]
    held = list(gc.overworld_potions)
    assert held[0] == before[0]
    assert held[1] == offers[0]
    assert held[2:] == before[2:]


def test_card_reveal_last_and_skip_keeps_bowl_native_semantics(reward_env):
    env = reward_env
    advance_to(env, 'card')
    assert env._obs()['offer_card_ids'].any()
    assert not env._obs()['reward_potion_ids'].any()
    env.thick.gc.set_relics(['Singing Bowl'])
    hp = env.thick.gc.max_hp
    count = env.thick.gc.get_reward_info()['card_count']
    env.step(A._RSKIP)
    assert env.thick.gc.max_hp == hp + 2
    assert env.thick.gc.get_reward_info()['card_count'] == count - 1
    assert stage(env) == 'done'
    assert np.flatnonzero(env.action_masks()).tolist() == [A._PROCEED]
    env.step(A._PROCEED)
    assert env._phase() != 'combat_reward'
