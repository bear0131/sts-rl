"""H1/V1 use the same real sampled transitions and differ only in leaf values."""
import json

import pytest
import slaythespire as sts

import env.search.turn_planner as module
from env.core.combat_value import combat_terminal_target
from env.core.native_actions import complete_actions
from env.search.turn_planner import NativeTurnPlanner, TurnPlannerConfig, power_horizon

pytestmark = [pytest.mark.native, pytest.mark.timeout(60)]
END = 4 << 29


def planner(**kwargs):
    return NativeTurnPlanner(TurnPlannerConfig(leaf_boundary='next-decision', **kwargs))


def snapshot(battle):
    return json.dumps(battle.to_dict(), sort_keys=True)


def setup(combat_factory, kind):
    if kind == 'headbutt':
        battle = combat_factory(['HEADBUTT'] * 5).thick.bc
        battle.apply_custom_state({'hand': ['HEADBUTT'], 'draw': ['DEFEND_RED'] * 5,
                                   'discard': ['WOUND', 'CARNAGE'], 'exhaust': []})
        battle.execute(sts.Action(sts.ActionType.CARD, 0, 0))
    elif kind == 'codex':
        battle = combat_factory(['DEFEND_RED'] * 10, relics=["Nilry's Codex"]).thick.bc
        battle.apply_custom_state({'player': {'energy': 0}})
    elif kind == 'time-warp':
        battle = combat_factory(['POMMEL_STRIKE'] * 25, encounter='Time Eater',
                                relics=["Nilry's Codex"]).thick.bc
        battle.apply_custom_state({'player': {'energy': 20}, 'hand': ['POMMEL_STRIKE'] * 5,
                                   'draw': ['POMMEL_STRIKE'] * 20, 'discard': []})
        for _ in range(11):
            battle.execute(sts.Action(sts.ActionType.CARD, 0, 0))
        assert ('Time Warp', 11) in battle.to_dict()['monsters'][0]['powers']
    elif kind == 'darkling':
        battle = combat_factory(['DEFEND_RED'] * 15, encounter='Three Darklings').thick.bc
        battle.apply_custom_state({'player': {'hp': 80, 'block': 999, 'energy': 3},
            'hand': ['STRIKE_RED'], 'draw': ['DEFEND_RED'] * 15, 'discard': [],
            'monsters': [{}, {'hp': 6, 'block': 0}, {}]})
        battle.execute(sts.Action(sts.ActionType.CARD, 0, 1))
        killed = battle.to_dict()
        assert killed['outcome'] == 0 and killed['monsters'][1]['half_dead']
        assert killed['monsters'][1]['move'] == 'DARKLING_REGROW'
        battle.execute(sts.Action(sts.ActionType.END_TURN))
        waiting = battle.to_dict()
        assert waiting['outcome'] == 0 and waiting['monsters'][1]['half_dead']
        assert waiting['monsters'][1]['move'] == 'DARKLING_REINCARNATE'
    else:
        battle = combat_factory(['STRIKE_RED'] * 10, encounter='Awakened One').thick.bc
        battle.apply_custom_state({'hand': ['STRIKE_RED'], 'draw': ['DEFEND_RED'] * 5,
            'discard': [], 'monsters': [{'hp': 0}, {'hp': 0}, {'hp': 6, 'block': 0}]})
        battle.execute(sts.Action(sts.ActionType.CARD, 0, 2))
        assert battle.to_dict()['monsters'][2]['half_dead']
    return battle


@pytest.mark.parametrize('kind', ['headbutt', 'codex', 'time-warp', 'rebirth', 'darkling'])
def test_h1_v1_same_traversal_and_live_leaves(combat_factory, monkeypatch, kind):
    battle = setup(combat_factory, kind)
    before = snapshot(battle)
    pristine = battle.clone()
    original_leaf = module.leaf_value
    h1_leaves, v1_leaves, traces = [], [], []
    def capture_leaf(raw, **kwargs):
        if raw['outcome'] == 0:
            h1_leaves.append(json.dumps(raw, sort_keys=True))
        return original_leaf(raw, **kwargs)
    monkeypatch.setattr(module, 'leaf_value', capture_leaf)
    class Counted:
        def __init__(self, inner): self.inner = inner
        def __getattr__(self, name): return getattr(self.inner, name)
        def clone(self): return Counted(self.inner.clone())
        def execute(self, action):
            self.inner.execute(action)
            traces.append(snapshot(self.inner))
        def execute_to_enemy_boundary(self, action):
            pytest.fail('next-decision must never project an enemy boundary')
    p = planner(worlds=2, node_budget=512)
    sample = p.sampler.clone_and_redeterminize
    monkeypatch.setattr(p.sampler, 'clone_and_redeterminize', lambda *a, **kw: Counted(sample(*a, **kw)))
    h1 = p.choose(battle, state_nonce=17, public_relics=())
    h1_trace = list(traces)
    traces.clear()
    batches = []
    def evaluate(states):
        batches.append(len(states))
        for state in states:
            raw = state.to_dict()
            assert raw['outcome'] == 0 and not raw['post_enemy_turn_boundary']
            assert raw['turn'] > battle.turn
            v1_leaves.append(snapshot(state))
        return [.4] * len(states)
    v1 = p.choose(battle, state_nonce=17, public_relics=(), leaf_evaluator=evaluate)
    assert h1_trace == traces and len(traces) == v1.diagnostics['native_transitions']
    assert h1_leaves == v1_leaves and v1_leaves
    assert sum(batches) == v1.diagnostics['model_leaf_evaluations'] == len(v1_leaves)
    assert len(batches) == v1.diagnostics['model_leaf_batches']
    assert max(batches) <= 64
    for key in ('native_transitions', 'expanded_nodes', 'fallback_leaves', 'max_action_depth',
                'second_turn_entries', 'completed_horizon_leaves', 'budget_exhausted', 'depth_limit_hits'):
        assert h1.diagnostics[key] == v1.diagnostics[key]
    assert v1.diagnostics['native_transitions'] <= v1.diagnostics['effective_total_budget']
    assert v1.diagnostics['max_action_depth'] <= 16
    assert all('model_value_mean' in o and 'heuristic_mean' not in o
               for o in v1.diagnostics['root_outcomes'].values())
    assert snapshot(battle) == before
    if kind == 'headbutt':
        assert {'WOUND', 'CARNAGE'} <= {json.loads(s)['hand'][0][0] for s in v1_leaves}
    if kind == 'rebirth':
        assert all(json.loads(s)['monsters'][2]['hp'] == 300 for s in v1_leaves)
    if kind == 'darkling':
        for state in map(json.loads, v1_leaves):
            assert [m['id'] for m in state['monsters']] == ['DARKLING'] * 3
            middle = state['monsters'][1]
            assert middle['hp'] == middle['max_hp'] // 2
            assert not middle['half_dead'] and not middle['gone']
    # Root RNG/callback preservation includes actual future draws and intents,
    # which are not proven merely by equality of the current to_dict snapshot.
    for _ in range(4):
        legal = complete_actions(pristine)
        if not legal:
            break
        action = END if END in legal else legal[0]
        pristine.execute(sts.Action(action))
        battle.execute(sts.Action(action))
        assert snapshot(battle) == snapshot(pristine)


@pytest.mark.parametrize('monster_hp', [40, 30000])
def test_model_batches_are_bounded_even_with_many_complete_leaves(combat_factory, monster_hp):
    battle = combat_factory(['ANGER'] * 10).thick.bc
    battle.apply_custom_state({'hand': ['ANGER'] * 8, 'draw': ['WOUND'] * 5,
                               'monsters': [{'max_hp': monster_hp}]})
    # Native custom-state HP is first clamped against the existing maximum.
    battle.apply_custom_state({'monsters': [{'hp': monster_hp}]})
    assert battle.to_dict()['monsters'][0]['hp'] == monster_hp
    sizes = []
    result = planner(worlds=1, node_budget=8192, beam_width=32, max_depth=8).choose(
        battle, state_nonce=9, leaf_evaluator=lambda states: sizes.append(len(states)) or [.3]*len(states))
    assert len(sizes) > result.diagnostics['root_actions']
    assert max(sizes) == 64 and min(sizes) > 0
    assert sum(sizes) == result.diagnostics['model_leaf_evaluations']
    outcomes = result.diagnostics['root_outcomes']
    assert outcomes[str(END)]['live_horizon_steps_mean'] == 1
    if monster_hp == 30000:
        assert result.action_bits == END
        assert all(outcomes[str(i)]['live_horizon_steps_mean'] == 2 for i in range(8))
    else:
        assert result.action_bits == 0
        assert all(outcomes[str(i)]['wins'] == 1 and outcomes[str(i)]['live_horizon_steps_total'] == 0
                   and outcomes[str(i)]['live_horizon_steps_mean'] is None for i in range(8))


def test_best_live_depth_resets_for_a_later_higher_value(combat_factory):
    battle = combat_factory(['WOUND'] * 10).thick.bc
    battle.apply_custom_state({'hand': ['DEFEND_RED', 'STRIKE_RED'],
        'draw': ['WOUND'] * 5, 'discard': [],
        'player': {'hp': 80, 'energy': 3},
        'monsters': [{'hp': 40, 'block': 0, 'move': 'JAW_WORM_CHOMP'}]})
    before = snapshot(battle)
    seen_hp = []
    def evaluate(states):
        hp = [s.to_dict()['monsters'][0]['hp'] for s in states]
        seen_hp.extend(hp)
        return [.6 if value < 40 else .5 for value in hp]
    value, counts = planner()._root(battle, 0, 100, lambda: None, (),
        power_horizon(battle.to_dict()), battle.turn, evaluate)
    assert seen_hp == [40, 34]  # Defend -> END, then Defend -> Strike -> END.
    assert value == (0, .6) and counts['live_steps'] == 3
    assert counts['winning_steps'] == counts['fallback_leaves'] == 0
    assert snapshot(battle) == before


def test_pending_depth_fallback_uses_model_scale_and_no_extra_execute(combat_factory):
    battle = setup(combat_factory, 'headbutt')
    seen = []
    result = planner(worlds=2, node_budget=1, max_depth=1).choose(
        battle, state_nonce=23,
        leaf_evaluator=lambda states: seen.extend(snapshot(s) for s in states) or [.17]*len(states))
    d = result.diagnostics
    assert d['max_action_depth'] == 1
    assert d['native_transitions'] == d['root_actions'] * 2
    assert d['completed_horizon_leaves'] == 0
    assert d['depth_limit_hits'] > 0
    assert d['fallback_leaves'] == d['root_actions'] * 2 == len(seen)
    assert set(d['root_values'].values()) == {.17}
    assert all(o['unresolved'] == 2 and o['boundary_live'] == 0 for o in d['root_outcomes'].values())


@pytest.mark.parametrize('kind', ['win', 'loss'])
def test_real_terminal_uses_exact_shared_target_without_callback(combat_factory, kind):
    battle = combat_factory(['STRIKE_RED'] * 5).thick.bc
    battle.apply_custom_state({'hand': ['STRIKE_RED'],
        'player': {'hp': 40 if kind == 'win' else 1, 'energy': 3},
        'monsters': [{'hp': 6 if kind == 'win' else 40, 'block': 0, 'move': 'JAW_WORM_CHOMP'}]})
    action = 0 if kind == 'win' else END
    expected = battle.clone()
    expected.execute(sts.Action(action))
    assert expected.to_dict()['outcome'] == (1 if kind == 'win' else 2)
    value, counts = planner()._root(battle, action, 2, lambda: None, (), power_horizon(battle.to_dict()),
        battle.turn, lambda states: pytest.fail('terminal must not call model'))
    assert value == (1 if kind == 'win' else -1, combat_terminal_target(expected))
    assert counts['model_leaf_evaluations'] == 0
    assert counts['winning_steps'] == (1 if kind == 'win' else 0)


@pytest.mark.parametrize('bad', [float('nan'), float('inf'), -.01, 1.01])
def test_model_rejects_nonfinite_or_out_of_target_range(combat_factory, bad):
    battle = combat_factory(['DEFEND_RED'] * 10).thick.bc
    with pytest.raises(ValueError, match='finite.*\\[0, 1\\]'):
        planner(worlds=1, node_budget=32).choose(battle, state_nonce=1,
                                              leaf_evaluator=lambda states: [bad]*len(states))


def test_model_rejects_wrong_number_of_values(combat_factory):
    battle = combat_factory(['DEFEND_RED'] * 10).thick.bc
    with pytest.raises(ValueError, match='value count'):
        planner(worlds=1, node_budget=32).choose(battle, state_nonce=1, leaf_evaluator=lambda states: [])


def test_projected_boundary_and_invalid_combinations_rejected(combat_factory):
    with pytest.raises(ValueError, match='lookahead_turns'):
        TurnPlannerConfig(leaf_boundary='next-decision', lookahead_turns=2)
    with pytest.raises(ValueError, match='leaf_boundary'):
        TurnPlannerConfig(leaf_boundary='unknown')
    battle = combat_factory(['DEFEND_RED'] * 10).thick.bc
    with pytest.raises(ValueError, match='next-decision'):
        NativeTurnPlanner().choose(battle, state_nonce=1, leaf_evaluator=lambda states: [])
    battle.execute_to_enemy_boundary(END)
    with pytest.raises(ValueError, match='live.*decision'):
        planner().choose(battle, state_nonce=1)
    pending = combat_factory(['DEFEND_RED'] * 10, relics=["Nilry's Codex"]).thick.bc
    pending.execute_to_enemy_boundary(END)
    before = snapshot(pending)
    with pytest.raises(ValueError, match='projected enemy boundary'):
        planner().choose(pending, state_nonce=1, leaf_evaluator=lambda states: pytest.fail('projected model input'))
    assert snapshot(pending) == before


def test_delayed_live_batch_does_not_replace_win_or_its_shortest_depth(combat_factory):
    battle = combat_factory(['STRIKE_RED'] * 10).thick.bc
    battle.apply_custom_state({'player': {'hp': 40, 'energy': 3},
        'hand': ['DEFEND_RED', 'DEFEND_RED', 'STRIKE_RED'],
        'monsters': [{'hp': 6, 'block': 0, 'move': 'JAW_WORM_CHOMP'}]})
    seen = []
    value, counts = planner(max_depth=4)._root(battle, 0, 100, lambda: None, (),
        power_horizon(battle.to_dict()), battle.turn,
        lambda states: seen.extend(states) or [.99]*len(states))
    direct = battle.clone()
    direct.execute(sts.Action(sts.ActionType.CARD, 0, 0))
    direct.execute(sts.Action(sts.ActionType.CARD, 1, 0))
    assert seen and counts['model_leaf_evaluations'] == len(seen)
    assert value == (1, combat_terminal_target(direct))
    assert counts['winning_steps'] == 2
    assert counts['live_steps'] == 0


def test_equal_terminal_model_values_keep_shortest_winning_sequence(combat_factory):
    battle = combat_factory(['STRIKE_RED'] * 10, encounter='Chosen', relics=['Medical Kit']).thick.bc
    battle.apply_custom_state({'monsters': [{'move': 'CHOSEN_HEX'}]})
    battle.execute(sts.Action(sts.ActionType.END_TURN))
    battle.apply_custom_state({'player': {'hp': 70, 'energy': 3,
        'powers': [('DARK_EMBRACE', 1), ('COMBUST', 5)]},
        'hand': ['DAZED'] * 8, 'draw': ['DAZED'] * 2, 'discard': [], 'exhaust': [],
        'monsters': [{'hp': 2, 'block': 0}]})
    result = planner(worlds=2, node_budget=2048).choose(battle, state_nonce=2375,
        public_relics=('Medical Kit',), leaf_evaluator=lambda states: [.5]*len(states))
    assert result.action_bits == END
    outcomes = result.diagnostics['root_outcomes']
    assert outcomes[str(END)]['winning_steps_mean'] == 1
    assert all(outcomes[str(i)]['winning_steps_mean'] == 2 for i in range(8))
    assert len(set(result.diagnostics['root_values'].values())) == 1


@pytest.mark.parametrize('escaped,outcome', [(True, 0), (False, 99)])
def test_unsupported_terminal_is_explicit_not_a_loss(combat_factory, escaped, outcome):
    battle = combat_factory(['DEFEND_RED'] * 10).thick.bc
    # Synthetic metadata isolates rejection; this is not evidence the pinned
    # native build implements Smoke Bomb escape.
    class Terminal:
        def clone(self): return self
        def execute(self, action): pass
        def to_dict(self):
            raw = battle.to_dict()
            return dict(raw, escaped=escaped, outcome=outcome)
    with pytest.raises(ValueError, match='unknown or escaped'):
        planner()._root(Terminal(), END, 2, lambda: None, (), power_horizon(battle.to_dict()),
                        battle.turn, lambda states: pytest.fail('unknown terminal reached model'))
