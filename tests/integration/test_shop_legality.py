import pytest

from env.core import game_actions as ga
from env.core.game_env import SimGameEnv


def _reach_shop():
    env = SimGameEnv()
    env.reset(seed=0)
    env.gc.skip_battles = True
    for _ in range(200):
        if env.phase() == "shop":
            return env
        actions = env.legal_semantic_actions()
        assert actions
        env.step(ga.proceed() if ga.proceed() in actions else actions[0])
    raise AssertionError("fixed seed/path did not reach a shop")


def _potion_options(gc):
    return [option for option in gc.legal_shop_options() if option[0] == "potion"]


def _reach_potion_reward():
    env = SimGameEnv()
    env.reset(seed=0)
    env.gc.skip_battles = True
    for _ in range(100):
        if env.phase() == "combat_reward" and env.gc.get_reward_info()["potions"]:
            return env
        actions = env.legal_semantic_actions()
        assert actions
        env.step(ga.proceed() if ga.proceed() in actions else actions[0])
    raise AssertionError("fixed seed/path did not reach a potion reward")


def _reward_potion_options(gc):
    return [option for option in gc.legal_reward_options() if option[0] == "potion"]


def test_shop_potions_require_capacity_and_no_sozu():
    env = _reach_shop()
    gc = env.gc
    gc.gold = 999
    gc.set_potion_state([])
    assert _potion_options(gc)

    potion_idx = _potion_options(gc)[0][1]
    gc.set_relics(["Sozu"])
    assert not _potion_options(gc)
    before = (gc.gold, list(gc.overworld_potions))
    with pytest.raises(ValueError, match="shop_buy_potion requires a legal action"):
        gc.shop_buy_potion(potion_idx)
    assert (gc.gold, list(gc.overworld_potions)) == before

    gc.set_relics([])
    gc.set_potion_state(["Fire Potion", "Blood Potion", "Block Potion"])
    assert not _potion_options(gc)
    before = (gc.gold, list(gc.overworld_potions))
    with pytest.raises(ValueError, match="shop_buy_potion requires a legal action"):
        gc.shop_buy_potion(potion_idx)
    assert (gc.gold, list(gc.overworld_potions)) == before


def test_reward_potions_require_capacity_and_no_sozu_without_losing_offer():
    env = _reach_potion_reward()
    gc = env.gc
    offered = list(gc.get_reward_info()["potions"])
    assert offered
    assert _reward_potion_options(gc)
    assert ga.claim(ga.CLAIM_POTION, 0) in env.legal_semantic_actions()

    gc.set_relics(["Sozu"])
    assert not _reward_potion_options(gc)
    assert ga.claim(ga.CLAIM_POTION, 0) not in env.legal_semantic_actions()
    with pytest.raises(ValueError, match="claim_potion requires a legal action"):
        gc.claim_potion(0)
    assert list(gc.get_reward_info()["potions"]) == offered

    gc.set_relics([])
    gc.set_potion_state(["Fire Potion", "Blood Potion", "Block Potion"])
    assert not _reward_potion_options(gc)
    assert ga.claim(ga.CLAIM_POTION, 0) not in env.legal_semantic_actions()
    with pytest.raises(ValueError, match="claim_potion requires a legal action"):
        gc.claim_potion(0)
    assert list(gc.get_reward_info()["potions"]) == offered
