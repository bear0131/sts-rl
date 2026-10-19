"""Real native counterfactual branches, arbitrary strategy names and F03 limits."""
import json

import numpy as np
import slaythespire as sts

from env.core import action_space as A
from env.core.native_actions import complete_actions
from rl.evaluation.baselines.combat_audit import CombatAuditStrategy
from rl.evaluation.strategies import Decision, DecisionRequest, strategy_manifest


class CompleteFirst:
    name = 'arbitrary-complete-strategy'
    combat_action_contract = 'native-v1'

    def decide(self, requests, *, check):
        return [Decision(None, self.name, native_action=complete_actions(r.env.thick.bc)[0])
                for r in requests]


class LegacyFirst:
    name = 'arbitrary-legacy-strategy'  # no contract declaration means legacy

    def decide(self, requests, *, check):
        assert all(r.env.combat_action_contract == 'legacy' for r in requests)
        result = []
        for request in requests:
            legal = list(np.flatnonzero(request.env.action_masks()))
            result.append(Decision(int(next((a for a in legal if a != A.END_TURN), legal[0])), self.name))
        return result


def request(env):
    return DecisionRequest(env, env._obs(), env._info(), 42, 17)


def prepare(env, *, selection_open, padding=10):
    env.combat_action_contract = 'native-v1'
    env.thick.bc.apply_custom_state({'player': {'energy': 3, 'strength': 999},
        'hand': ['SECRET_TECHNIQUE'], 'draw': ['STRIKE_RED'] * padding + ['DEFEND_RED', 'SHRUG_IT_OFF'],
        'discard': [], 'exhaust': []})
    env.thick.bc.set_player_strength(999)
    if selection_open:
        env.thick.bc.execute(sts.Action(sts.ActionType.CARD, 0))
    env._invalidate_backend_cache()


def test_native_trace_uses_decision_action_with_no_thin_ranking(combat_factory):
    env = combat_factory(['SECRET_TECHNIQUE'] * 5)
    prepare(env, selection_open=True, padding=96)
    assert not env.action_masks().any()
    before = json.dumps(env.thick.bc.to_dict(), sort_keys=True)
    trace = []
    audit = CombatAuditStrategy(CompleteFirst(), trace=trace.append)
    chosen = audit.decide([request(env)], check=lambda: None)[0]
    assert chosen.native_action == (2 << 29 | 96)
    event = trace[0]
    assert event['native_action'] == dict(bits=chosen.native_action,
        kind='SINGLE_CARD_SELECT', source=96, target=0)
    assert event['rule_action'] is None and event['ranked'] == []
    assert json.dumps(env.thick.bc.to_dict(), sort_keys=True) == before
    env.step_native_combat(chosen.native_action)
    assert env.thick.bc.to_dict()['hand'][0][0] == 'DEFEND_RED'
    assert strategy_manifest(audit)['reference']['combat_action_contract'] == 'native-v1'


def test_each_fork_owns_contract_and_late_selection_remains_actionable(combat_factory):
    env = combat_factory(['SECRET_TECHNIQUE'] * 5)
    prepare(env, selection_open=True)
    before = json.dumps(env.thick.bc.to_dict(), sort_keys=True)
    rows, trace = [], []
    audit = CombatAuditStrategy(CompleteFirst(), candidates={'old': LegacyFirst(), 'new': CompleteFirst()},
        limit=1, rows=rows, trace=trace.append, max_steps=50, timeout=10)
    audit.decide([request(env)], check=lambda: None)
    old, new = rows
    assert old['result'] == 'win' and old['steps'] > 0
    assert old['terminated'] and not old['truncated']
    assert old['candidate_combat_action_contract'] == 'legacy'
    assert new['candidate_combat_action_contract'] == 'native-v1'
    assert new['result'] == 'win' and new['terminated'] and not new['truncated']
    assert old['reference_combat_action_contract'] == new['reference_combat_action_contract'] == 'native-v1'
    assert old['visible_state_sha256'] == new['visible_state_sha256']
    assert env.combat_action_contract == 'native-v1'
    assert json.dumps(env.thick.bc.to_dict(), sort_keys=True) == before


def test_late_selection_entered_during_shared_runner_completes(combat_factory):
    env = combat_factory(['SECRET_TECHNIQUE'] * 5)
    prepare(env, selection_open=False)
    rows = []
    CombatAuditStrategy(CompleteFirst(), candidates={'old': LegacyFirst()}, limit=1,
        rows=rows, max_steps=20, timeout=10).decide([request(env)], check=lambda: None)
    assert rows[0]['result'] == 'win'
    assert rows[0]['steps'] > 0
    assert rows[0]['terminated'] and not rows[0]['truncated']
    assert 'initial_dead_screen' not in rows[0]
