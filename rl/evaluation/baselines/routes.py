"""Shared deterministic map objective; random tie-breaking belongs to the strategy."""
from functools import lru_cache
from typing import Callable
from env.core.spiremap import MAX_Y

def _route_values(rmap) -> Callable[[int, int], tuple[int, int]]:
    """Return a cached (elite_count, rest_count) value for each map node."""

    @lru_cache(maxsize=None)
    def value(x: int, y: int) -> tuple[int, int]:
        if y > MAX_Y:
            return 0, 0
        room = rmap.room_type(x, y)
        own_elites = int(room == "ELITE")
        own_rests = int(room == "REST")
        if y == MAX_Y:
            return own_elites, own_rests
        children = rmap.reachable(x, y)
        if not children:
            return own_elites, own_rests
        child_elites, child_rests = min(
            (value(child, y + 1) for child in children),
            key=lambda score: (score[0], -score[1]),
        )
        return own_elites + child_elites, own_rests + child_rests

    return value


def optimal_route_choices(
    choices: list[tuple[int, int]],
    rmap,
    current_y: int,
) -> tuple[list[int], tuple[int, int]]:
    """Return all lexicographically optimal legal actions and their value."""
    next_y = 0 if current_y < 0 else current_y + 1
    value = _route_values(rmap)
    scored = [
        ((0, 0) if next_y > MAX_Y else value(column, next_y), action_id)
        for column, action_id in choices
    ]
    best_score = min(
        (score for score, _action_id in scored),
        key=lambda score: (score[0], -score[1]),
    )
    optimal = [action_id for score, action_id in scored if score == best_score]
    return optimal, best_score
