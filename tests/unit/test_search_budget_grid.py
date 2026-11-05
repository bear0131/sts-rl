import json

import numpy as np
import pytest

from tools.analysis.search_budget_grid import compare, load_games, ranking


def test_complete_games_are_paired_by_actual_seed_and_repeated_controls(tmp_path):
    a = dict(
        config=["uct", 1, 1, -1],
        terminals=[[1, 55], [0, 9]],
        counts=[400, 100],
        wins=1,
        work=[50, 12500],
    )
    b = dict(
        config=["flat", 1, 2, 0.005],
        terminals=[[1, 55], [1, 54]],
        counts=[400, 371],
        wins=2,
        work=[65, 16250],
    )
    panel = dict(
        actual_game_ids=[23991, 1827],
        weights="upper",
        lower="lower",
        rank=0,
        wall_seconds=4.0,
        arms=[a, a, b],
    )
    (tmp_path / "panel-0.json").write_text(json.dumps(panel))
    games, records = load_games(tmp_path)
    first, second = ranking(records, records)
    assert list(games) == panel["actual_game_ids"] and first[0] == "flat"
    draws = np.array([[0, 1], [1, 0]])
    delta = compare(records, first, second, draws)
    assert delta["a_wins_only"] == 1 and delta["b_wins_only"] == 0
    assert delta["win_difference"] == 0.5 and delta["floor_difference"] == 22.5

    more = tmp_path / "panel-1.json"
    more.write_text(json.dumps(dict(panel, actual_game_ids=[1827, 23991])))
    with pytest.raises(ValueError, match="same games"):
        load_games(tmp_path)
    more.write_text(json.dumps(dict(panel, arms=[dict(a, counts=[400, 101])])))
    with pytest.raises(ValueError, match="repeated control differs"):
        load_games(tmp_path)
