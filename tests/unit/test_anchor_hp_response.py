from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import torch

from env.core.encode import HP_SCALE
from rl.evaluation.anchor_readiness import anchor_hp_response


def test_hp_response_changes_only_current_hp_and_ratios_without_mutation():
    obs = dict(combat_exit_anchor=np.array([1., 0.]),
        player_scalars=np.array([25/HP_SCALE, 80/HP_SCALE, .1, .3, .4, 25/80], dtype=np.float32),
        run_hp=np.array([25/HP_SCALE, 80/HP_SCALE, 25/80], dtype=np.float32),
        deck_ids=np.array([1, 2]), potions=np.array([3, 0]), relic_counters=np.array([5.]))
    before = deepcopy(obs)
    class Policy:
        def predict_values(self, batch):
            for name in ('combat_exit_anchor', 'deck_ids', 'potions', 'relic_counters'):
                np.testing.assert_array_equal(batch[name].numpy(), np.repeat(obs[name][None], len(batch[name]), axis=0))
            np.testing.assert_array_equal(batch['player_scalars'][:, 1:5], np.repeat(obs['player_scalars'][None, 1:5], len(batch['run_hp']), axis=0))
            torch.testing.assert_close(batch['player_scalars'][:, 0], batch['run_hp'][:, 0])
            torch.testing.assert_close(batch['player_scalars'][:, 5], batch['run_hp'][:, 2])
            torch.testing.assert_close(batch['run_hp'][:, 2], batch['run_hp'][:, 0] * HP_SCALE / 80)
            return batch['player_scalars'][:, :1] * 2
    result = anchor_hp_response(SimpleNamespace(policy=Policy(), device='cpu'), obs, hp=25, max_hp=80)
    assert result['hp_levels'][0] == 1 and result['hp_levels'][-1] == 80
    assert {20, 24, 25, 26, 30} <= set(result['hp_levels'])
    assert result['actual_value'] == .25
    assert (np.diff(result['values']) > 0).all()
    for key in before:
        np.testing.assert_array_equal(obs[key], before[key])
