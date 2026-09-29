"""F05: learners and searches value the fight before post-combat healing."""
from types import SimpleNamespace

import pytest

from env.core.combat_value import combat_terminal_target, VALUE_CONTRACT
from rl.algorithms.accumulating_ppo import _fill_nosl_combat_value_targets
from rl.algorithms.nosl_replay import NOSLISAZReplayBuffer
from env.search.nosl_is_az import _terminal_value
from rl.search.nosl_search_methods import _terminal_hp
from rl.search.combat_value_search import CombatValueSearch


@pytest.mark.parametrize("hp,expected_exit", [(50, 56), (78, 80)])
def test_real_settlement_heals_but_training_target_does_not(
    short_combat_env_type, hp, expected_exit
):
    env = short_combat_env_type()
    try:
        env.reset(seed=42)
        battle = env.thick.bc
        battle.apply_custom_state({
            "player": {"hp": hp, "energy": 3, "powers": [("STRENGTH", 999)]},
            "hand": ["WHIRLWIND"],
        })
        env._invalidate_backend_cache()
        branch = env.fork_combat()
        branch._backend_step_native_combat(0)
        assert branch.thick.bc.is_battle_over
        assert _terminal_value(branch) == pytest.approx(hp / 200)
        assert _terminal_hp(branch) == hp
        assert CombatValueSearch._terminal_hp_value(
            SimpleNamespace(env=branch)
        ) == pytest.approx(hp / 200)
        _, _, _, info = env._backend_step_native_combat(0)
        assert env.thick.bc is None
        assert env.thick.gc.cur_hp == expected_exit
        assert info["nosl_value_terminal_target"] == pytest.approx(hp / 200)
        samples = [{"value_target": None}, {"value_target": None}]
        _fill_nosl_combat_value_targets(samples, info["nosl_value_terminal_target"])
        assert all(s["value_target"] == pytest.approx(hp / 200) for s in samples)
    finally:
        env.close()


def test_loss_value_is_zero_even_with_nonzero_hp_metadata():
    assert combat_terminal_target({"outcome": 2, "player": {"hp": 7}}) == 0


def test_replay_rejects_old_or_different_value_labels(combat_factory):
    env = combat_factory(["STRIKE_RED"] * 5)
    replay = NOSLISAZReplayBuffer(
        capacity=2, observation_space=env.observation_space,
        observation_keys=["player_scalars"], action_dim=env.action_space.n,
    )
    state = replay.state_dict()
    assert state["value_contract"] == VALUE_CONTRACT
    for contract in (None, "legacy-boss-transition"):
        wrong = dict(state, value_contract=contract)
        with pytest.raises(ValueError, match="value contract mismatch"):
            replay.load_state_dict(wrong)
