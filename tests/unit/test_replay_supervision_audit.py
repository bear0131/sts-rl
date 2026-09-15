"""Encoded enemy strata are proxies, not reconstructed trajectory metadata."""
import numpy as np
import pytest

from tools.analysis.probe_nosl_updates import supervision_strata, run_replay_audit


def test_replay_strata_cover_live_rows_and_require_enemy_presence_and_hp():
    monsters = np.zeros((5, 2, 3), dtype=np.float16)
    monsters[0, 0, :2] = [1, .1]
    monsters[1, 0, :2] = [1, .1]
    monsters[2, 0, :2] = [0, .1]  # padding with stale HP is not alive
    monsters[3, 0, :2] = [1, 0]  # present but defeated
    replay = dict(size=4, observations=dict(
        player_scalars=np.array([[1/200], [5/200], [20/200], [0], [80/200]], dtype=np.float16),
        monster_scalars=monsters), arrays=dict(policy_weights=np.array([1, 0, 0, 0, 1])))
    hp, strata = supervision_strata(replay)
    np.testing.assert_array_equal(hp, [1, 5, 20, 0])
    populated = {(kind, interval): rows.tolist() for kind, interval, rows in strata if len(rows)}
    assert populated == {('policy_root', '1-5'): [0],
        ('value_only_living_enemy', '1-5'): [1],
        ('value_only_no_living_enemy', '6-20'): [2],
        ('value_only_no_living_enemy', 'dead'): [3]}


@pytest.mark.parametrize('budget', [0, -1, float('nan'), float('inf')])
def test_read_only_audit_rejects_invalid_budget_before_io(tmp_path, budget):
    with pytest.raises(ValueError, match='budgets'):
        run_replay_audit(tmp_path/'absent', tmp_path/'result.json', max_seconds=budget)
