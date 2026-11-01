"""Censoring discards whole games, not just their unfinished last decision."""
from types import SimpleNamespace
import numpy as np
from rl.training.episode_rollout import EpisodeRolloutConfig, exclude_samples
import pytest


def test_censor_excludes_all_previous_fights_and_strategic_decisions_of_that_game():
    # Earlier completed fights from the censored game cannot leak into CE/PPO.
    retained_upper={'env':0,'episode':8,'reward':1.}
    retained_lower={'env':0,'recommended_action':12}
    model=SimpleNamespace(_upper_completed=[{'env':1,'episode':8,'reward':.5},retained_upper,
                                            {'env':1,'episode':8,'reward':.7}],
        _upper_pending=[None,{'reward':.2}],_upper_censored=0,
        _small_playout_samples=[{'env':1,'recommended_action':2},retained_lower,{'env':1,'recommended_action':3}])
    exclude_samples(model,SimpleNamespace(episode_rollout_censored=np.array([False,True])))
    assert model._upper_completed==[retained_upper]
    assert model._small_playout_samples==[retained_lower]
    assert model._upper_pending==[None,None]
    assert model._cohort_excluded_samples=={'excluded_upper_samples':3,'excluded_lower_samples':2}


def test_budget_contract_rejects_unknown_censor_policy():
    with pytest.raises(ValueError,match='budget'):
        EpisodeRolloutConfig(240,20261101,censored_games='count-as-death')
