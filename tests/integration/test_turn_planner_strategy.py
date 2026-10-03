"""Public-information and shared-runner contracts for the planner adapter."""
from dataclasses import asdict
import json
from types import SimpleNamespace

import numpy as np
import pytest
import slaythespire as sts

from env.core import action_space as A, combat_env
from rl.evaluation.baselines.turn_planner import RiskTurnPlannerStrategy, TurnPlannerStrategy
from env.search.turn_planner import TurnPlannerConfig
from rl.evaluation.runner import EvaluationConfig, Scenario, evaluate
from rl.evaluation.strategies import DecisionRequest, PhaseStrategy, strategy_manifest


def _request(env):
    return DecisionRequest(env, env._obs(), env._info(), 42, 0)


def _native(env):
    # Shared fixture already creates a real native combat; opt into its complete
    # action executor before any test transition (legacy remains the default).
    env.combat_action_contract = 'native-v1'
    return env


def _forbidden(*args, **kwargs):
    pytest.fail('Runic Dome must never enter a planner with the hidden live intent')


def test_runic_dome_uses_same_visible_rule_action_without_calling_planner(
        combat_factory, monkeypatch):
    env = _native(combat_factory(['STRIKE_RED'] * 5, relics=['Runic Dome']))
    strategy = TurnPlannerStrategy(trace=(trace := []).append)
    monkeypatch.setattr(strategy.planner, 'choose', _forbidden)
    decisions = []
    for move in ('JAW_WORM_CHOMP', 'JAW_WORM_BELLOW'):
        env.thick.bc.apply_custom_state({
            'hand': ['STRIKE_RED', 'DEFEND_RED'], 'monsters': [{'move': move}]})
        env._invalidate_backend_cache()
        decisions.append(strategy.decide([_request(env)])[0])
    assert decisions[0] == decisions[1]
    assert decisions[0].diagnostics['planner_dome_fallback'] == 1
    assert decisions[0].native_action is None
    assert env.action_masks()[decisions[0].action]
    assert all(row['state']['monsters'] == [] for row in trace)
    json.dumps(trace)  # Persistable diagnostic payload, with no raw hidden intent.


@pytest.mark.parametrize('choices', [
    ['DEFEND_RED', 'SHRUG_IT_OFF'], ['SHRUG_IT_OFF', 'DEFEND_RED']])
def test_runic_dome_late_draw_choice_works_without_planner_or_legacy_mask(
        combat_factory, monkeypatch, choices):
    env = _native(combat_factory(['SECRET_TECHNIQUE'] * 5, relics=['Runic Dome']))
    env.thick.bc.apply_custom_state({
        'hand': ['SECRET_TECHNIQUE'],
        'draw': ['STRIKE_RED'] * 12 + choices,
        'discard': [], 'exhaust': [],
    })
    env._invalidate_backend_cache()
    env.step_native_combat(0)
    # The standalone 71-action mask cannot represent native draw indices 12/13.
    # ACTION_VERSION 7 maps the whole-run mask through the public candidate list.
    assert not np.any(combat_env.legal_mask(env.thick.bc))
    descriptor = env.thick.select_descriptor()
    assert descriptor['zone'] == 'draw'
    assert {candidate['native_index'] for candidate in descriptor['candidates']} == {12, 13}
    assert [candidate['card']['id'] for candidate in descriptor['candidates']] == ['Defend_R', 'Shrug It Off']
    np.testing.assert_array_equal(np.flatnonzero(env.action_masks()), [A._PICK, A._PICK + 1])
    strategy = TurnPlannerStrategy()
    monkeypatch.setattr(strategy.planner, 'choose', _forbidden)
    decision = strategy.decide([_request(env)])[0]
    assert decision.action == A._PICK + 1
    assert decision.native_action is None
    assert decision.diagnostics['planner_dome_fallback'] == 1
    oracle = env.thick.bc.clone()
    oracle.execute(sts.Action(sts.ActionType.SINGLE_CARD_SELECT, 12 + choices.index('SHRUG_IT_OFF')))
    _, _, _, truncated, info = env.step(decision.action)
    assert not truncated
    assert not info.get('illegal')
    assert env.thick.bc.to_dict() == oracle.to_dict()
    assert env.thick.select_descriptor() is None


@pytest.mark.parametrize('context', ['CODEX', 'DISCOVERY'])
def test_runic_dome_generated_choices_keep_native_handles_and_codex_skip(
        combat_factory, monkeypatch, context):
    from rl.evaluation.baselines.combat_rules import selection_score, visible_combat

    relics = ['Runic Dome'] + (["Nilry's Codex"] if context == 'CODEX' else [])
    env = _native(combat_factory(['DISCOVERY'] * 5, relics=relics))
    action = sts.ActionType.END_TURN if context == 'CODEX' else sts.ActionType.CARD
    env.thick.bc.execute(sts.Action(action, 0))
    env._invalidate_backend_cache()
    descriptor = env.thick.select_descriptor()
    assert descriptor['context'] == context and descriptor['zone'] == 'generated'
    assert len(env.thick.bc.select_info()['generated_cards']) == 3
    count = 4 if context == 'CODEX' else 3
    assert [candidate['native_index'] for candidate in descriptor['candidates']] == list(range(count))
    np.testing.assert_array_equal(np.flatnonzero(env.action_masks()), A._PICK + np.arange(count))
    view = visible_combat(_request(env))
    for index, candidate in enumerate(descriptor['candidates'][:3]):
        assert candidate['card']['id'] == view.piles['generated'][index]['id']
        assert np.isfinite(selection_score(index, view)[0])
    if context == 'CODEX':
        assert descriptor['candidates'][3]['card'] is None
        assert selection_score(3, view) == (2., {'skip_generated': 2.})
        oracle = env.thick.bc.clone()
        oracle.execute(sts.Action(sts.ActionType.SINGLE_CARD_SELECT, 3))
        branch = env.fork_combat()
        try:
            branch.step(A._PICK + 3)
            assert branch.thick.bc.to_dict() == oracle.to_dict()
        finally:
            branch.close()
    strategy = TurnPlannerStrategy()
    monkeypatch.setattr(strategy.planner, 'choose', _forbidden)
    decision = strategy.decide([_request(env)])[0]
    assert decision.native_action is None and env.action_masks()[decision.action]
    assert decision.diagnostics['planner_dome_fallback'] == 1
    native_index = descriptor['candidates'][decision.action - A._PICK]['native_index']
    oracle = env.thick.bc.clone()
    oracle.execute(sts.Action(sts.ActionType.SINGLE_CARD_SELECT, native_index))
    env.step(decision.action)
    assert env.thick.bc.to_dict() == oracle.to_dict()
    assert env.thick.select_descriptor() is None


def test_composed_manifest_keeps_native_contract_and_exact_planner_config():
    class Upper:
        name = 'fixed-upper-sentinel'
    config = TurnPlannerConfig(worlds=3, node_budget=37, beam_width=5,
                               max_depth=7, master_nonce=987654)
    strategy = TurnPlannerStrategy(config)
    manifest = strategy_manifest(PhaseStrategy(Upper(), {'combat': strategy}))
    assert manifest['default']['name'] == Upper.name
    combat = manifest['phases']['combat']
    assert combat['combat_action_contract'] == 'native-v1'
    assert combat['config'] == asdict(config)
    assert combat['version'] == strategy.version
    assert combat['information'] == 'independent-NOSL-futures'
    assert combat['dome'] == 'visible-rule-fallback'
    assert combat['fallback']['name'] == strategy.fallback.name
    json.dumps(manifest)


def test_shared_runner_executes_complete_target_and_preserves_rich_trace(combat_factory):
    env = _native(combat_factory(['STRIKE_RED'] * 5, encounter='Three Sentries'))
    env.thick.bc.apply_custom_state({
        'hand': ['STRIKE_RED'],
        'monsters': [{'hp': 0}, {'hp': 6, 'block': 0}, {'hp': 0}],
    })
    env._invalidate_backend_cache()
    branch = env.fork_combat()
    trace, progress = [], []
    strategy = TurnPlannerStrategy(TurnPlannerConfig(worlds=2, node_budget=32),
                                   trace=trace.append)
    scenario = Scenario('planner-native-fixture', lambda: branch,
        outcome=lambda e, i, t, tr: dict(
            result='win' if int(e.thick.bc.outcome) == 1 else 'death', floor=1),
        reset=lambda e, seed: (e._obs(), e._info()))
    rows = list(evaluate(strategy, scenario,
        EvaluationConfig(episodes=1, seed=42, max_steps=3, timeout=10),
        progress=progress.append))
    assert len(rows) == 1 and rows[0]['result'] == 'win'
    assert rows[0]['steps'] == 1
    assert rows[0]['combat_action_contract'] == 'native-v1'
    assert rows[0]['search_decisions'] == 1
    assert rows[0]['worlds_completed'] == 2
    assert 'root_values' not in rows[0]  # Structured data belongs in trace, not numeric sums.
    assert 'max_action_depth' not in rows[0]  # Maxima must not become misleading runner sums.
    assert trace[0]['action'] is None
    action = sts.Action(trace[0]['native_action'])
    assert action.action_type == sts.ActionType.CARD
    assert action.source_idx == 0 and action.target_idx == 1
    assert isinstance(trace[0]['diagnostics']['root_values'], dict)
    assert trace[0]['diagnostics']['max_action_depth'] >= 1
    assert progress[0]['current_native_action'] == trace[0]['native_action']
    json.dumps({'rows': rows, 'trace': trace, 'progress': progress})


def test_wrapper_rejects_legacy_before_invoking_planner(combat_factory, monkeypatch):
    env = combat_factory(['STRIKE_RED'] * 5)
    assert env.combat_action_contract == 'legacy'
    strategy = TurnPlannerStrategy()
    monkeypatch.setattr(strategy.planner, 'choose',
                        lambda *a, **kw: pytest.fail('legacy environment reached planner'))
    with pytest.raises(ValueError, match='native-v1'):
        strategy.decide([_request(env)])


def test_cli_composition_preserves_planner_budget_and_supplied_upper():
    from rl.evaluation.eval import build_parser
    from rl.evaluation.strategies import compose_strategy
    upper = object()  # No model or checkpoint is loaded for a parser/dispatch contract.
    args = build_parser().parse_args([
        'unused-checkpoint.zip', '--solver', 'turn-planner',
        '--planner-worlds', '3', '--planner-nodes', '97',
        '--planner-beam', '4', '--planner-depth', '9', '--planner-nonce', '777',
    ])
    strategy = compose_strategy(upper, args)
    assert strategy.default.model is upper
    manifest = strategy_manifest(strategy)['phases']['combat']
    assert manifest['combat_action_contract'] == 'native-v1'
    assert manifest['config'] == dict(
        worlds=3, node_budget=97, beam_width=4, max_depth=9, master_nonce=777,
        lookahead_turns=1, leaf_profile='time-clock-v9')


@pytest.mark.parametrize('room,encounter,selected', [
    ('BOSS', 'SLIME_BOSS', True), ('BOSS', 'AWAKENED_ONE', True),
    ('BOSS', 'DONU_AND_DECA', True), ('BOSS', 'TIME_EATER', True),
    ('ELITE', 'GREMLIN_NOB', True), ('ELITE', 'LAGAVULIN', True),
    ('ELITE', 'NEMESIS', True), ('ELITE', 'THREE_SENTRIES', False),
    ('EVENT', 'GREMLIN_NOB', False), ('EVENT', 'LAGAVULIN_EVENT', False),
    ('MONSTER', 'NEMESIS', False), ('BOSS_TREASURE', 'TIME_EATER', False),
])
def test_risk_route_uses_only_public_room_and_frozen_encounters(room, encounter, selected):
    # No battle, intent, observation or model is available to this predicate.
    gc = SimpleNamespace(cur_room=getattr(sts.Room, room),
                         encounter=getattr(sts.MonsterEncounter, encounter))
    env = SimpleNamespace(unwrapped=SimpleNamespace(thick=SimpleNamespace(gc=gc)))
    assert RiskTurnPlannerStrategy.is_risk_combat(SimpleNamespace(env=env)) is selected


def test_risk_cli_manifest_mixed_native_batch_and_both_traces(combat_factory, monkeypatch):
    from rl.evaluation.eval import build_parser
    from rl.evaluation.strategies import compose_strategy
    args = build_parser().parse_args([
        'unused.zip', '--task', 'wholerun', '--solver', 'turn-planner',
        '--planner-turns', '2', '--planner-turn-scope', 'boss-elite-v1',
        '--planner-nodes', '64', '--planner-worlds', '2', '--planner-depth', '6'])
    trace = []
    composed = compose_strategy(object(), args, trace=trace.append)
    strategy = composed.routes['combat']
    manifest = strategy_manifest(composed)['phases']['combat']
    assert manifest['turn_scope'] == 'boss-elite-v1'
    assert manifest['predicate'] == dict(
        source='public-gc.cur_room-and-gc.encounter', all_rooms=['BOSS'],
        encounter_room='ELITE', encounters=['GREMLIN_NOB', 'LAGAVULIN', 'NEMESIS'])
    one = manifest['default']['config']
    two = manifest['phases']['combat']['config']
    assert one == dict(two, lookahead_turns=1)
    assert two['lookahead_turns'] == 2 and two['node_budget'] == 64 and two['max_depth'] == 6
    assert composed.combat_action_contract == 'native-v1'
    json.dumps(manifest)
    requests, originals, expected = [], [], []
    for step, room in enumerate((sts.Room.BOSS, sts.Room.MONSTER, sts.Room.BOSS)):
        env = combat_factory(['DEFEND_RED'] * 5, combat_action_contract='native-v1')
        env.thick.gc.cur_room = room
        env.thick.bc.apply_custom_state({'hand': ['DEFEND_RED'], 'draw': ['STRIKE_RED'] * 5,
            'discard': [], 'monsters': [{'hp': 40, 'move': 'JAW_WORM_CHOMP'}]})
        env._invalidate_backend_cache()
        request = DecisionRequest(env, env._obs(), env._info(), 42, step)
        requests.append(request)
        originals.append(env.thick.bc.to_dict())
        config = TurnPlannerConfig(**(one if room == sts.Room.MONSTER else two))
        expected.append(TurnPlannerStrategy(config).decide([request])[0])
    calls = {1: [], 2: []}
    for turns, child in ((1, strategy.default), (2, strategy.routes['combat'])):
        original_decide = child.decide
        def counted(batch, *, check, turns=turns, original_decide=original_decide):
            calls[turns].extend(r.step for r in batch)
            return original_decide(batch, check=check)
        monkeypatch.setattr(child, 'decide', counted)
    actual = composed.decide(requests)
    assert [d.native_action for d in actual] == [d.native_action for d in expected]
    assert calls == {1: [1], 2: [0, 2]}
    assert len(trace) == 3
    by_step = {row['step']: row for row in trace}
    for i, decision in enumerate(actual):
        turns = 1 if i == 1 else 2
        assert decision.diagnostics[f'planner_turn{turns}_decisions'] == 1
        assert decision.diagnostics[f'planner_turn{3 - turns}_decisions'] == 0
        assert by_step[i]['lookahead_turns'] == turns
        assert by_step[i]['turn2_applied'] is (turns == 2)
        assert by_step[i]['diagnostics']['effective_node_budget'] == 64
        assert by_step[i]['diagnostics']['effective_total_budget'] == 128
        assert decision.diagnostics['native_transitions'] <= 128
        assert requests[i].env.thick.bc.to_dict() == originals[i]
    json.dumps(trace)


def test_risk_dome_trace_distinguishes_chosen_horizon_from_search(combat_factory, monkeypatch):
    env = combat_factory(['STRIKE_RED'] * 5, relics=['Runic Dome'],
                         combat_action_contract='native-v1')
    env.thick.gc.cur_room = sts.Room.BOSS
    trace = []
    strategy = RiskTurnPlannerStrategy(TurnPlannerConfig(lookahead_turns=2), trace=trace.append)
    for child in strategy._unique_strategies():
        monkeypatch.setattr(child.planner, 'choose', _forbidden)
    decision = strategy.decide([_request(env)])[0]
    assert decision.diagnostics['planner_dome_fallback'] == 1
    assert decision.diagnostics['planner_turn1_decisions'] == 0
    assert decision.diagnostics['planner_turn2_decisions'] == 0
    assert decision.diagnostics['effective_total_budget'] == 0
    assert decision.diagnostics['native_transitions'] == 0
    assert trace[0]['lookahead_turns'] == 2 and trace[0]['turn2_applied'] is False
    assert trace[0]['state']['monsters'] == []


@pytest.mark.parametrize('options', [
    ['--solver', 'policy', '--planner-turns', '2'],
    ['--solver', 'policy', '--planner-turns', '2', '--combat-search-depth', '1'],
    ['--solver', 'turn-planner', '--planner-turns', '1'],
    ['--solver', 'turn-planner', '--planner-turns', '2', '--task', 'combat'],
])
def test_invalid_risk_cli_scope_fails_before_checkpoint_load(monkeypatch, options):
    import sys
    from rl.evaluation import eval as cli
    monkeypatch.setattr(sys, 'argv', ['eval', 'unused.zip', '--planner-turn-scope',
                                     'boss-elite-v1', *options])
    monkeypatch.setattr(cli, 'load_checkpoint', lambda *a, **kw: pytest.fail('loaded checkpoint'))
    with pytest.raises(ValueError, match='boss-elite-v1 requires'):
        cli.main()


def test_risk_cli_cannot_be_rewritten_to_legacy_value_search(monkeypatch):
    import sys
    from rl.evaluation import eval as cli
    monkeypatch.setattr(sys, 'argv', ['eval', 'unused.zip', '--solver', 'turn-planner',
        '--planner-turn-scope', 'boss-elite-v1', '--planner-turns', '2',
        '--combat-search-depth', '1'])
    monkeypatch.setattr(cli, 'load_checkpoint', lambda *a, **kw: pytest.fail('loaded checkpoint'))
    with pytest.raises(ValueError, match='combat-search cannot be combined'):
        cli.main()
