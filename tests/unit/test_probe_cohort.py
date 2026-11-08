from types import SimpleNamespace
import pytest

from rl.algorithms.upper_probe import probe_cohort


def cohort():
    model = SimpleNamespace(_upper_completed=[dict(env=i, episode=2, terminal=True, reward=0.) for i in [0, 2, 3]],
        _small_playout_samples=[dict(env=i) for i in [0, 2, 2, 3]],
        independent_rollout_config=dict(max_combat_turns=50),
        episode_rollout_stats=dict(games=4, completed_games=3, censored_games=1, unfinished_games=0, first_seed=20),
        episode_rollout_censored_slots=[dict(slot=1, seed=21, reason='combat-turn-limit:51>50')])
    protocol = dict(diagnostic=dict(games_per_rank=4, holdout_games_per_rank=1, allow_combat_turn_censoring=True))
    return model, protocol


def test_only_exact_production_retained_games_are_accepted():
    model, protocol = cohort()
    result = probe_cohort(model, protocol)
    assert result['retained_games'] == 3 and result['requested_games'] == 4
    assert result['censored_slots'] == model.episode_rollout_censored_slots
    model.episode_rollout_censored_slots[0]['reason'] = 'combat-turn-limit:53>50'
    assert probe_cohort(model, protocol)['retained_games'] == 3
    protocol['diagnostic']['allow_combat_turn_censoring'] = False
    with pytest.raises(ValueError, match='every complete game'):
        probe_cohort(model, protocol)


@pytest.mark.parametrize('mutation,reason', [
    ('lower_leak', 'leaked'), ('upper_leak', 'upper records'), ('unknown_reason', 'censor reason'),
    ('wrong_count', 'accounting mismatch'), ('unfinished', 'accounting mismatch'),
    ('wrong_seed', 'censor reason'), ('duplicate_slot', 'accounting mismatch')])
def test_incomplete_or_leaked_cohorts_rejected_collectively(mutation, reason):
    model, protocol = cohort()
    if mutation == 'lower_leak':model._small_playout_samples.append(dict(env=1))
    elif mutation == 'upper_leak':model._upper_completed[0]['env'] = 1
    elif mutation == 'unknown_reason':model.episode_rollout_censored_slots[0]['reason'] = 'wall-clock-budget'
    elif mutation == 'wrong_count':model.episode_rollout_stats['completed_games'] = 4
    elif mutation == 'unfinished':model.episode_rollout_stats['unfinished_games'] = 1
    elif mutation == 'wrong_seed':model.episode_rollout_censored_slots[0]['seed'] = 99
    else:model.episode_rollout_censored_slots *= 2
    votes = []
    def vote(value):votes.append(value);return value
    with pytest.raises(ValueError, match=reason):probe_cohort(model, protocol, average=vote)
    assert votes == [dict(probe_cohort_valid=0.)]


def test_valid_rank_also_exits_if_another_rank_rejects():
    model, protocol = cohort()
    with pytest.raises(ValueError, match='another rank'):
        probe_cohort(model, protocol, average=lambda value: dict(probe_cohort_valid=.75))
