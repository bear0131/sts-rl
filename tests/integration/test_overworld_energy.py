"""Actual native payment and Wound semantics behind the upper prior."""
from copy import deepcopy

from env.core import action_space as A
from rl.evaluation.baselines.overworld_energy import EnergyAwareUpperStrategy, energy_profile
from rl.evaluation.baselines.overworld_rules import OverworldRuleStrategy
from rl.evaluation.strategies import DecisionRequest
from env.core.run_env import SlayTheSpireEnv


def test_three_to_four_energy_allows_two_two_cost_cards(combat_factory):
    for relics, expected_plays in (((), 1), (('Sozu',), 2)):
        env = combat_factory(['FLAME_BARRIER'] * 5, relics=relics)
        assert env.thick.bc.to_dict()['player']['energy'] == 3 + bool(relics)
        for _ in range(expected_plays):
            assert env.action_masks()[A._PLAY]
            env.step(A._PLAY)
            if env._info()["pending"]:
                legal = [i for i, allowed in enumerate(env.action_masks()) if allowed]
                assert len(legal) == 1
                env.step(legal[0])
        raw = env.thick.bc.to_dict()
        assert raw['player']['block'] == 12 * expected_plays
        assert not env.action_masks()[A._PLAY]


def test_mark_of_pain_native_draws_initial_hand_before_two_wounds(combat_factory):
    env = combat_factory(['FLAME_BARRIER'] * 5, relics=['Mark of Pain', 'Medical Kit'])
    raw = env.thick.bc.to_dict()
    assert raw['player']['energy'] == 4
    assert len(raw['hand']) == 5 and all(c[0] == 'FLAME_BARRIER' for c in raw['hand'])
    assert len(raw['draw']) == 2 and all(c[0] == 'WOUND' for c in raw['draw'])
    public = env.thick.obs_canonical()
    before = deepcopy(public)
    profile = energy_profile(public)
    assert profile['energy'] == 4
    assert len(public['global']['deck']) == 5
    assert len(profile['cards']) == 7  # include the owned relic's transient draw tax.
    assert env.thick.obs_canonical() == before


def test_energy_upper_preserves_actual_neow_choice_and_state():
    env = SlayTheSpireEnv()
    try:
        observation, info = env.reset(seed=10000)
        before = deepcopy(env.thick.obs_canonical())
        request = DecisionRequest(env, observation, info, 10000, 0)
        actual = EnergyAwareUpperStrategy().decide([request])[0]
        expected = OverworldRuleStrategy().decide([request])[0]
        assert actual == expected and env.action_masks()[actual.action]
        assert env.thick.obs_canonical() == before
        env.step(actual.action)
    finally:
        env.close()
