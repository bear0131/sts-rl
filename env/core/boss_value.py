"""Optional act-boss terminal value objective.

The normal combat objective is deliberately kept outside this module.  These
helpers describe the opt-in objective only, using primitive values so the same
rules can be used by the Python environment, search adapters, and tests without
importing the simulator extension.
"""

from __future__ import annotations

import math


HP_SCALE = 200.0


def is_boss_room(room) -> bool:
    """Public room identity; stable through splits, summons and resurrection.

    Includes both A20 bosses and the Heart. The transition-healing experiment
    below has different eligibility and must not be reused for this objective.
    """
    return enum_name(room).split('.')[-1] in {'BOSS', 'MONSTERROOMBOSS'}


def win_hp_utility(win, hp, weight):
    """Expectation or observed label: win * (weight + (1-weight) * hp).

    hp is capped pre-settlement HP / 200 conditional on winning. Inputs may
    be scalars or tensors; callers own outcome validation and the boss mask.
    """
    return win * (weight + (1.0 - weight) * hp)


def enum_name(value) -> str:
    """Return a stable tail name for strings and pybind enum values."""
    if value is None:
        return ""
    return str(value).split(".")[-1].upper()


def is_act_boss(
    *,
    cur_room,
    act: int,
    ascension: int,
    encounter,
    boss,
    second_boss=None,
    floor_num: int | None = None,
) -> bool:
    """Whether the live combat is one of the three ordinary act bosses.

    ``cur_room`` is required in addition to the encounter.  Event combats and
    test fixtures can force a boss encounter while remaining in another room.
    At A20 the first Act 3 boss is excluded exactly as the simulator's
    ``GameContext::afterBattle`` branch excludes it; the second boss remains
    eligible.
    """
    if enum_name(cur_room) != "BOSS":
        return False
    act = int(act)
    if not 1 <= act <= 3:
        return False
    encounter_name = enum_name(encounter)
    boss_name = enum_name(boss)
    if act in (1, 2):
        return encounter_name == boss_name
    if encounter_name == boss_name:
        # At A20 this is the first Act 3 boss and the simulator immediately
        # enters the second boss instead of transitioning acts.
        return int(ascension) < 20
    second_name = enum_name(second_boss)
    if second_name:
        return int(ascension) >= 20 and encounter_name == second_name
    # Older Python extensions did not expose second_boss.  The simulator's
    # second Act 3 boss is entered on floor 51, so retain an exact fallback for
    # that build rather than treating arbitrary forced encounters as bosses.
    return int(ascension) >= 20 and floor_num == 51


def _round_half_up(value: float) -> int:
    # C++ std::round is used by GameContext::transitionToAct.  All values here
    # are non-negative, so floor(x + .5) matches it without Python's bankers'
    # rounding.
    return int(math.floor(float(value) + 0.5))


def post_boss_hp(
    *,
    act: int,
    ascension: int,
    max_hp: int,
    exit_hp: int,
    face_of_cleric: bool = False,
) -> int:
    """HP after the combat exit and the relevant act transition.

    Act 3 victory is the requested full-heal objective.  For A5+ Act 1/2 the
    simulator restores 75% of missing HP; lower ascensions are fully healed.
    ``exit_hp`` is expected to include combat-exit relic healing.
    """
    maximum = max(0, int(max_hp)) + (1 if face_of_cleric else 0)
    hp = min(maximum, max(0, int(exit_hp)))
    act = int(act)
    if act == 3 or int(ascension) < 5:
        return maximum
    if act in (1, 2):
        return min(maximum, hp + _round_half_up((maximum - hp) * 0.75))
    return hp


def terminal_target(
    *,
    enabled: bool,
    eligible: bool,
    outcome,
    act: int,
    ascension: int,
    max_hp: int,
    exit_hp: int,
    face_of_cleric: bool = False,
) -> float | None:
    """Return the normalized opt-in terminal label, or ``None`` if unchanged.

    A single observed win is labelled ``H_max / 200`` (with the Act 1/2 A5+
    transition convention); a loss is labelled zero.  Averaging these labels
    estimates ``H_max / 200 * P(win)`` without pretending the per-trajectory
    win probability is known.
    """
    if not enabled or not eligible:
        return None
    name = enum_name(outcome)
    if name in {"PLAYER_LOSS", "LOSS", "2"} or outcome == 2:
        return 0.0
    if not (name in {"PLAYER_VICTORY", "VICTORY", "WIN", "1"} or outcome == 1):
        return None
    hp = post_boss_hp(
        act=act,
        ascension=ascension,
        max_hp=max_hp,
        exit_hp=exit_hp,
        face_of_cleric=face_of_cleric,
    )
    return float(min(1.0, max(0.0, hp / HP_SCALE)))


def mode_for_act_boss(*, act: int, ascension: int) -> int:
    """Native-search terminal mode: 1=fixed/full, 2=A5+ Act1/2 heal."""
    return 2 if int(ascension) >= 5 and int(act) in (1, 2) else 1


def context_value_mode(*, enabled: bool, gc, bc) -> int:
    """Return the native terminal mode for one live combat root."""
    if not enabled or gc is None or bc is None:
        return 0
    act = int(getattr(gc, "act", 0))
    ascension = int(getattr(gc, "ascension", 0))
    if not is_act_boss(
        cur_room=getattr(gc, "cur_room", None),
        act=act,
        ascension=ascension,
        encounter=getattr(gc, "encounter", getattr(bc, "encounter", None)),
        boss=getattr(gc, "boss", None),
        second_boss=getattr(gc, "second_boss", None),
        floor_num=getattr(gc, "floor_num", getattr(bc, "floor_num", None)),
    ):
        return 0
    return mode_for_act_boss(act=act, ascension=ascension)


def _face_of_cleric(gc) -> bool:
    relic_ids = getattr(gc, "relic_ids", None)
    if not callable(relic_ids):
        return False
    try:
        relics = relic_ids()
    except (AttributeError, RuntimeError, TypeError):
        return False
    return any(
        enum_name(relic).replace("_", "").replace(" ", "")
        == "FACEOFCLERIC"
        for relic in relics
    )


def context_terminal_target(*, enabled: bool, gc, bc) -> float | None:
    """Compute an opt-in target from live Game/BattleContext objects.

    This adapter intentionally uses only public binding attributes, keeping
    eligibility and transition arithmetic identical for env and Python search.
    ``None`` means the legacy target must be retained.
    """
    if not enabled or gc is None or bc is None:
        return None
    state = bc.to_dict() if hasattr(bc, "to_dict") else {}
    player = state.get("player") or {}
    combat_player = getattr(bc, "player", None)
    act = int(getattr(gc, "act", 0))
    ascension = int(getattr(gc, "ascension", 0))
    eligible = context_value_mode(enabled=True, gc=gc, bc=bc) != 0
    outcome = state.get("outcome", getattr(bc, "outcome", None))
    exit_hp = player.get("hp", getattr(combat_player, "curHp", 0))
    if enum_name(outcome) in {"PLAYER_VICTORY", "VICTORY", "WIN", "1"} or outcome == 1:
        projected_exit_hp = getattr(bc, "projected_exit_hp", None)
        if callable(projected_exit_hp):
            try:
                exit_hp = projected_exit_hp(gc)
            except (AttributeError, RuntimeError, TypeError):
                # Legacy/fake contexts may not support projection.  Retain the
                # current-HP fallback so disabled and compatibility paths keep
                # their existing behavior.
                pass
    return terminal_target(
        enabled=True,
        eligible=eligible,
        outcome=outcome,
        act=act,
        ascension=ascension,
        max_hp=player.get("max_hp", getattr(combat_player, "maxHp", 0)),
        exit_hp=exit_hp,
        face_of_cleric=_face_of_cleric(gc),
    )
