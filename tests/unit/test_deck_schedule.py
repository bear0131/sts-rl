import json
from copy import deepcopy
from types import SimpleNamespace
from dataclasses import asdict
import pytest

from rl.algorithms.deck_schedule import DeckTrainingSchedule, configure, check_due, advance, apply_stage


def healthy_moments():
    return dict(natural=64., censored=0., anchors=128., target_sum=64., target_sq_sum=64.,
                error_sum=0., error_sq_sum=1., relic_offered=30., relic_taken=25., floor_sum=512.,
                wins=2., timeout=0., failed=0.)


def test_qualified_search_covers_all_roots_while_only_loss_weight_ramps():
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
    for completed, beta in ((180, 0.), (190, .05), (200, .1)):
        apply_stage(model, schedule, completed, rank=2)
        assert model.policy.combat_value_target == 'pre-reward-anchor'
        assert model.deck_search_config['fraction'] == 1.
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


def test_filtered_objective_can_explicitly_continue_old_search_state_without_reset():
    model = SimpleNamespace(policy=SimpleNamespace(core=SimpleNamespace(model_contract='deck-latent-combat-v1'),
        combat_value_target='pre-reward-anchor'), policy_kwargs={},
        deck_training_state=dict(contract='deck-latent-stages-v1', stage='search', stage_start=200,
            good_checks=3, last_check=200, validation={'target_std': .328817}))
    before = deepcopy(model.deck_training_state)
    old = dict(search_qualified=True, search_backend='full480',
        search_action_selection='search-filter-policy-rank-v1', search_cuda_wave_capacity=65536)
    apply_stage(model, configure(model, old), 455)
    assert model.deck_search_config['objective'] == 'prior-advantage-kl-v1'
    apply_stage(model, configure(model, dict(old, search_objective='filtered-distillation-v1')), 455)
    assert model.deck_training_state == before
    assert model.deck_search_config['objective'] == 'filtered-distillation-v1'
    assert model.deck_search_config['normal_budget'] == 64 and model.deck_search_config['boss_budget'] == 256
    assert model.deck_search_config['coefficient'] == .1 and model.deck_search_config['fraction'] == 1.
    assert model.training_objective_contract == 'deck-latent-search-filter-distillation-v1'
    restored = deepcopy(model)
    apply_stage(restored, configure(restored), 456)
    assert restored.deck_search_config == model.deck_search_config
    assert restored.deck_training_state == before
    apply_stage(restored, configure(restored, dict(old, search_objective='turn-consensus-v1')), 455)
    assert restored.deck_training_state == before
    assert restored.deck_search_config['objective'] == 'turn-consensus-v1'
    assert restored.training_objective_contract == 'deck-latent-turn-consensus-v1'


@pytest.mark.parametrize('overrides,expected', [
    ({}, (64, 256)),
    ({'search_normal_budget': 256, 'search_boss_budget': 1024}, (256, 1024)),
])
def test_search_budgets_survive_collection_boundaries_and_saved_schedule(overrides, expected):
    model = SimpleNamespace(policy=SimpleNamespace(
        core=SimpleNamespace(model_contract='deck-latent-combat-v1'),
        combat_value_target='pre-reward-anchor'), policy_kwargs={},
        deck_training_state=dict(contract='deck-latent-stages-v1', stage='search',
            stage_start=240, validation=dict(target_std=.3)))
    schedule = configure(model, dict(search_qualified=True, search_backend='full480',
        search_action_selection='search-filter-policy-rank-v1', **overrides))
    apply_stage(model, schedule, 453, rank=2)
    restored = deepcopy(model)
    restored.deck_training_schedule = json.loads(json.dumps(model.deck_training_schedule))
    apply_stage(restored, configure(restored), 454, rank=2)
    for candidate in (model, restored):
        search = candidate.deck_search_config
        assert (search['normal_budget'], search['boss_budget']) == expected
        assert search['fraction'] == 1. and search['coefficient'] == .1
        assert search['target_std'] == .3
        assert search['action_selection'] == 'search-filter-policy-rank-v1'
    assert restored.deck_training_state == model.deck_training_state


@pytest.mark.parametrize('field', ['search_normal_budget', 'search_boss_budget'])
def test_search_schedule_rejects_empty_budget(field):
    with pytest.raises(ValueError, match='limits must be positive'):
        DeckTrainingSchedule(**{field: 0})
