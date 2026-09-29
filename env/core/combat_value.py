"""Fight-local value contract, captured before rewards and post-combat healing."""

from .encode import HP_SCALE


VALUE_CONTRACT = "combat-end-hp-v1"


def combat_terminal_outcome(battle):
    """Explicit outcome; an unknown/escaped/censored terminal is never a loss."""
    state = battle if isinstance(battle, dict) else battle.to_dict()
    if state.get('escaped', False):
        return 'escape'
    return {1:'win', 2:'loss'}.get(int(state['outcome']), 'unknown')


def combat_end_hp(battle) -> float:
    """Unscaled terminal HP; neither relic healing nor act transitions apply."""
    state = battle if isinstance(battle, dict) else battle.to_dict()
    if int(state["outcome"]) == 2:  # native Outcome::PLAYER_LOSS
        return 0.0
    return max(0.0, float(state["player"]["hp"]))


def combat_terminal_target(battle) -> float:
    return min(1.0, combat_end_hp(battle) / HP_SCALE)
