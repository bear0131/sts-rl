from copy import deepcopy
from types import SimpleNamespace
from dataclasses import asdict
import pytest

from rl.algorithms.deck_schedule import DeckTrainingSchedule, configure, check_due, advance, apply_stage


def healthy_moments():
    return dict(natural=64., censored=0., anchors=128., target_sum=64., target_sq_sum=64.,
                error_sum=0., error_sq_sum=1., relic_offered=30., relic_taken=25., floor_sum=512.,
                wins=2., timeout=0., failed=0.)


def test_three_validations_then_anchor_adaptation_and_qualified_search_ramp():
    model = SimpleNamespace(policy=SimpleNamespace(core=SimpleNamespace(model_contract='deck-latent-combat-v1'),
        combat_value_target='whole-run'), policy_kwargs={})
    schedule = configure(model, asdict(DeckTrainingSchedule()))
    state = model.deck_training_state
    for completed in (100, 120):
        assert check_due(state, schedule, completed)
        state = advance(state, schedule, completed, healthy_moments())
        assert state['stage'] == 'warmup'
    state = advance(state, schedule, 140, healthy_moments())
    assert state['stage'] == 'anchor'
    # An unqualified search runtime cannot silently activate after enough time.
    state = advance(state, schedule, 160, healthy_moments())
    assert state['stage'] == 'anchor'
    schedule = DeckTrainingSchedule(search_qualified=True)
    state = advance(state, schedule, 180, healthy_moments())
    assert state['stage'] == 'search'
    model.deck_training_state = deepcopy(state)
    for completed, coverage, beta in ((180, .25, 0.), (190, .5, .05), (200, 1., .1)):
        apply_stage(model, schedule, completed, rank=2)
        assert model.policy.combat_value_target == 'pre-reward-anchor'
        assert model.deck_search_config['fraction'] == coverage
        assert model.deck_search_config['coefficient'] == beta
    assert not check_due(state, schedule, 180)  # restart never repeats the same check


def test_censored_or_constant_future_targets_cannot_pass_anchor_gate():
    model = SimpleNamespace(policy=SimpleNamespace(core=SimpleNamespace(model_contract='deck-latent-combat-v1'),
        combat_value_target='whole-run'), policy_kwargs={})
    schedule = configure(model, asdict(DeckTrainingSchedule()))
    state = advance(model.deck_training_state, schedule, 100, healthy_moments())
    bad = healthy_moments(); bad.update(target_sq_sum=32.)  # constant target .5
    state = advance(state, schedule, 120, bad)
    assert state['good_checks'] == 0 and not state['validation']['passed']
    bad = healthy_moments(); bad.update(censored=10., natural=54.)
    state = advance(state, schedule, 140, bad)
    assert state['good_checks'] == 0 and state['stage'] == 'warmup'


def test_explicit_anchor_start_preserves_evidence_and_resume_adaptation():
    model = SimpleNamespace(policy=SimpleNamespace(core=SimpleNamespace(model_contract='deck-latent-combat-v1'),
        combat_value_target='whole-run'), policy_kwargs={})
    schedule = configure(model, asdict(DeckTrainingSchedule(anchor_start_round=101)))
    state = model.deck_training_state
    state['validation'] = {'passed': False, 'reason': 'operator decision remains separate'}
    evidence = deepcopy(state['validation'])
    apply_stage(model, schedule, 100)
    assert state['stage'] == 'warmup'
    apply_stage(model, schedule, 101)
    assert state['stage'] == 'anchor' and state['stage_start'] == 101
    assert model.policy.combat_value_target == 'pre-reward-anchor'
    assert state['good_checks'] == 0 and state['last_check'] == -1
    assert state['validation'] == evidence and state['transition_reason'] == 'configured-anchor-start'
    restored = deepcopy(model)
    apply_stage(restored, configure(restored), 140)
    assert restored.deck_training_state == state
    assert restored.deck_search_config is None
    with pytest.raises(ValueError, match='non-negative'):
        DeckTrainingSchedule(anchor_start_round=-1)
