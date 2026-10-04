"""Focused tests for the opt-in act-boss terminal-value helpers."""

from types import SimpleNamespace

import numpy as np
import pytest

from env.core.boss_value import (
    context_terminal_target,
    context_value_mode,
    is_act_boss,
    mode_for_act_boss,
    post_boss_hp,
    terminal_target,
)


def _boss_state(
    *,
    act=1,
    ascension=0,
    encounter="THE_GUARDIAN",
    boss="THE_GUARDIAN",
    second_boss=None,
    floor_num=None,
    room="BOSS",
):
    return dict(
        cur_room=room,
        act=act,
        ascension=ascension,
        encounter=encounter,
        boss=boss,
        second_boss=second_boss,
        floor_num=floor_num,
    )


def test_act_boss_requires_boss_room_and_matching_encounter():
    assert is_act_boss(**_boss_state())
    assert not is_act_boss(**_boss_state(room="MONSTER"))
    assert not is_act_boss(
        **_boss_state(encounter="JAW_WORM")
    )
    assert not is_act_boss(**_boss_state(act=4))


@pytest.mark.parametrize("act", [1, 2, 3])
def test_regular_act_bosses_are_eligible(act):
    assert is_act_boss(
        **_boss_state(act=act, encounter="BOSS_A", boss="BOSS_A")
    )


def test_a20_act3_first_boss_is_excluded():
    assert not is_act_boss(
        **_boss_state(
            act=3,
            ascension=20,
            encounter="AWAKENED_ONE",
            boss="AWAKENED_ONE",
            floor_num=50,
        )
    )


def test_a20_act3_second_boss_is_included_with_binding_metadata():
    assert is_act_boss(
        **_boss_state(
            act=3,
            ascension=20,
            encounter="TIME_EATER",
            boss="AWAKENED_ONE",
            second_boss="TIME_EATER",
            floor_num=51,
        )
    )


def test_a20_act3_second_boss_fallback_uses_floor_51():
    assert is_act_boss(
        **_boss_state(
            act=3,
            ascension=20,
            encounter="TIME_EATER",
            boss="AWAKENED_ONE",
            floor_num=51,
        )
    )
    assert not is_act_boss(
        **_boss_state(
            act=3,
            ascension=20,
            encounter="TIME_EATER",
            boss="AWAKENED_ONE",
            floor_num=50,
        )
    )


def test_post_boss_hp_matches_transition_healing_rules():
    # A0 transitions heal fully, while A5+ restores 75% of missing HP.
    assert post_boss_hp(act=1, ascension=0, max_hp=100, exit_hp=20) == 100
    assert post_boss_hp(act=2, ascension=5, max_hp=100, exit_hp=20) == 80
    assert post_boss_hp(act=1, ascension=5, max_hp=101, exit_hp=20) == 81
    # Act 3 victory enters the terminal/next-act path and is full HP here.
    assert post_boss_hp(act=3, ascension=20, max_hp=100, exit_hp=20) == 100
    assert post_boss_hp(
        act=1, ascension=0, max_hp=100, exit_hp=20, face_of_cleric=True
    ) == 101


def test_terminal_target_is_disabled_or_ineligible_without_override():
    kwargs = dict(
        outcome="PLAYER_VICTORY",
        act=1,
        ascension=0,
        max_hp=100,
        exit_hp=20,
    )
    assert terminal_target(enabled=False, eligible=True, **kwargs) is None
    assert terminal_target(enabled=True, eligible=False, **kwargs) is None


def test_terminal_target_labels_loss_zero_and_win_with_transition_hp():
    common = dict(
        enabled=True,
        eligible=True,
        act=1,
        ascension=0,
        max_hp=100,
        exit_hp=20,
    )
    assert terminal_target(outcome="PLAYER_LOSS", **common) == 0.0
    assert terminal_target(outcome="PLAYER_VICTORY", **common) == 0.5

    # The current helper deliberately follows the A5+ transition utility.
    assert terminal_target(
        outcome="PLAYER_VICTORY",
        act=1,
        ascension=5,
        max_hp=100,
        exit_hp=20,
        **{k: v for k, v in common.items() if k not in {"act", "ascension", "max_hp", "exit_hp"}},
    ) == 0.4


def test_terminal_target_returns_none_for_nonterminal_outcome_and_clips():
    common = dict(
        enabled=True,
        eligible=True,
        act=3,
        ascension=0,
        max_hp=250,
        exit_hp=1,
    )
    assert terminal_target(outcome="UNDECIDED", **common) is None
    assert terminal_target(outcome="PLAYER_VICTORY", **common) == 1.0


@pytest.mark.parametrize(
    ("act", "ascension", "expected"),
    [(1, 0, 1), (2, 4, 1), (1, 5, 2), (2, 20, 2), (3, 20, 1)],
)
def test_native_terminal_mode_matches_transition_family(act, ascension, expected):
    assert mode_for_act_boss(act=act, ascension=ascension) == expected


def test_context_value_mode_reuses_boss_eligibility():
    gc = SimpleNamespace(
        cur_room="BOSS",
        act=2,
        ascension=5,
        encounter="THE_CHAMP",
        boss="THE_CHAMP",
        second_boss=None,
        floor_num=33,
    )
    bc = SimpleNamespace(encounter="THE_CHAMP", floor_num=33)
    assert context_value_mode(enabled=False, gc=gc, bc=bc) == 0
    assert context_value_mode(enabled=True, gc=gc, bc=bc) == 2
    gc.cur_room = "MONSTER"
    assert context_value_mode(enabled=True, gc=gc, bc=bc) == 0


def test_context_terminal_target_uses_projected_combat_exit_hp():
    gc = SimpleNamespace(
        cur_room="BOSS",
        act=1,
        ascension=5,
        encounter="THE_GUARDIAN",
        boss="THE_GUARDIAN",
        second_boss=None,
        floor_num=16,
        relic_ids=lambda: [],
    )

    class Battle:
        encounter = "THE_GUARDIAN"
        floor_num = 16

        @staticmethod
        def to_dict():
            return {
                "outcome": "PLAYER_VICTORY",
                "player": {"hp": 20, "max_hp": 100},
            }

        @staticmethod
        def projected_exit_hp(projected_gc):
            assert projected_gc is gc
            return 32

    # 32 exit HP plus round(75% of the 68 missing HP) = 83.
    assert context_terminal_target(enabled=True, gc=gc, bc=Battle()) == 0.415


def test_opt_in_native_search_fails_closed_on_legacy_extension():
    from env.core.run_env import SlayTheSpireEnv

    class LegacyBattle:
        encounter = "THE_GUARDIAN"
        floor_num = 16

        @staticmethod
        def nosl_neural_search_begin(*_args):
            raise TypeError("incompatible function arguments")

    env = SlayTheSpireEnv.__new__(SlayTheSpireEnv)
    env._autoregressive_actions = False
    env.thick = SimpleNamespace(
        gc=SimpleNamespace(
            cur_room="BOSS",
            act=1,
            ascension=0,
            encounter="THE_GUARDIAN",
            boss="THE_GUARDIAN",
            second_boss=None,
            floor_num=16,
        ),
        bc=LegacyBattle(),
    )
    env._phase = lambda: "combat"

    with pytest.raises(RuntimeError, match="boss-transition terminal target"):
        env.nosl_is_az_cpp_begin(
            np.zeros(181),
            state_nonce=1,
            boss_transition_target=True,
        )


@pytest.mark.parametrize(
    ("payload", "expected"),
    [({}, False), ({"nosl_is_az_boss_transition_target": True}, True)],
)
def test_checkpoint_option_is_additive_and_legacy_safe(
    monkeypatch, payload, expected
):
    from stable_baselines3.common import save_util
    from rl.training.trainer import (
        _checkpoint_nosl_is_az_boss_transition_target,
    )

    monkeypatch.setattr(
        save_util,
        "load_from_zip_file",
        lambda *_args, **_kwargs: (dict(payload), {}, {}),
    )
    assert _checkpoint_nosl_is_az_boss_transition_target("unused.zip") is expected
