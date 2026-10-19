"""REWARD01: reward groups, capacity, and delayed selection retain identity."""
import subprocess

import pytest


@pytest.mark.timeout(240)
@pytest.mark.parametrize('ascension', [0, 15])
@pytest.mark.parametrize('variant', range(8), ids=['gold-indices', 'card-group-skip',
    'potion-capacity', 'potion-isolation', 'full-card-groups', 'bottle-return', 'relic-count', 'nested-relic-rewards'])
def test_native_reward_contract(game_context_executable, variant, ascension):
    executable, run_env = game_context_executable
    result = subprocess.run([str(executable), 'reward', str(variant), str(ascension)],
                            env=run_env, capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.endswith('PASS\n')


@pytest.fixture
def reward_env():
    import slaythespire as sts
    from env.core.run_env import SlayTheSpireEnv
    env = SlayTheSpireEnv()
    env.reset(seed=1)
    # These tests exercise reward transactions, not a model's map projection.
    # Reach the real combat through native semantic steps and its callbacks.
    for _ in range(64):
        if env.thick.bc is not None:
            break
        env.thick.step(env.thick.legal_semantic_actions()[0])
    assert env.thick.bc is not None
    env.thick.bc.set_player_strength(999)
    env.thick.bc.apply_custom_state({'hand': ['CLEAVE']})
    env.thick.bc.execute(sts.Action(sts.ActionType.CARD, 0, 0))
    env.thick._settle()
    env._invalidate_backend_cache()
    assert env.thick.phase() == 'combat_reward'
    assert env.thick.gc.get_reward_info()['potions'] == ['Block Potion']
    yield env
    env.close()


def test_sozu_masks_reward_without_consuming_it(reward_env):
    from env.core import game_actions as ga
    from env.core import action_space as A
    env = reward_env
    env.thick.gc.set_relics(['Sozu'])
    env._invalidate_backend_cache()
    assert ga.claim(ga.CLAIM_POTION, 0) not in env.thick.legal_semantic_actions()
    assert not env.action_masks()[A._RPOTION]


def test_sozu_direct_claim_is_rejected_without_losing_reward(reward_env):
    gc = reward_env.thick.gc
    gc.set_relics(['Sozu'])
    before = gc.get_reward_info()
    with pytest.raises(ValueError):
        gc.claim_potion(0)
    assert gc.get_reward_info() == before


def test_normal_claim_consumes_exactly_one_reward(reward_env):
    gc = reward_env.thick.gc
    gc.claim_potion(0)
    assert gc.get_reward_info()['potions'] == []
    assert list(gc.overworld_potions).count('Block Potion') == 1


@pytest.mark.parametrize('method', ['claim_gold', 'claim_potion', 'claim_relic'])
@pytest.mark.parametrize('index', [-1, 1, 256, 257])
def test_claim_rejects_missing_or_aliased_indices(reward_env, method, index):
    gc = reward_env.thick.gc
    before = (gc.gold, gc.get_reward_info(), list(gc.overworld_potions), gc.deck_ids())
    with pytest.raises(ValueError):
        getattr(gc, method)(index)
    assert (gc.gold, gc.get_reward_info(), list(gc.overworld_potions), gc.deck_ids()) == before


@pytest.mark.parametrize('screen', ['SHOP_ROOM', 'EVENT_SCREEN', 'MAP_SCREEN'])
@pytest.mark.parametrize('method', ['claim_gold', 'claim_potion', 'claim_relic', 'reward_proceed'])
def test_reward_bindings_reject_other_screens(reward_env, screen, method):
    import slaythespire as sts
    gc = reward_env.thick.gc
    gc.screen_state = getattr(sts.ScreenState, screen)
    before = (gc.gold, gc.get_reward_info(), gc.screen_state)
    assert gc.legal_reward_options() == []
    with pytest.raises(ValueError):
        getattr(gc, method)(*(() if method == 'reward_proceed' else (0,)))
    assert (gc.gold, gc.get_reward_info(), gc.screen_state) == before


def test_claim_consumed_gold_is_rejected_and_enumeration_has_item_indices(reward_env):
    gc = reward_env.thick.gc
    assert ('gold', 0, 0) in gc.legal_reward_options()
    assert ('potion', 0, 0) in gc.legal_reward_options()
    assert ('skip_card', 0, 5) in gc.legal_reward_options()
    amount = gc.get_reward_info()['gold'][0]
    before = gc.gold
    gc.claim_gold(0)
    assert gc.gold == before + amount
    assert ('gold', 0, 0) not in gc.legal_reward_options()
    with pytest.raises(ValueError):
        gc.claim_gold(0)
    assert gc.gold == before + amount
