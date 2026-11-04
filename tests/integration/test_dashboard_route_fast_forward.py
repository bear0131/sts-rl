from __future__ import annotations

from types import SimpleNamespace
import threading

import numpy as np
import pytest

from apps import dashboard
from env.core import game_actions as ga
from env.core.game_env import SimGameEnv


class _FirstLegalWatchSession(dashboard.WatchSession):
    """Deterministic outer policy for testing the Watch state machine."""

    def _prepare_decision(self):
        if self.terminated or self.truncated:
            self.decision = None
            return
        self.obs, self.info = self.env._obs(), self.env._info()
        legal = np.flatnonzero(self.env.action_masks())
        if not legal.size:
            self.decision = None
            return
        action = int(legal[0])
        self.decision = {
            "action": action,
            "label": self.env.describe_action(action),
            "value": None,
            "model_action": action,
            "search": None,
            "animation": self._action_animation(action),
            "top_actions": [],
        }


class _FixedModelOutputWatchSession(dashboard.WatchSession):
    """Exercise real upper routing without requiring a serialized policy."""

    def _policy_outputs(self, _observation, mask):
        legal = np.flatnonzero(mask)
        probs = np.zeros(len(mask), dtype=np.float32)
        probs[legal] = np.arange(1, len(legal) + 1, dtype=np.float32)
        probs /= probs.sum()
        return probs, 0.0, None


def _dummy_model():
    return SimpleNamespace(
        observation_space=SimpleNamespace(spaces={}),
        policy=SimpleNamespace(set_training_mode=lambda _enabled: None),
    )


@pytest.mark.parametrize('contract', ['native-v1', 'autoregressive-v2'])
def test_compact_mcts_budget_change_preserves_state_and_replays_native_action(contract):
    from rl.policies.small_playout import SmallPlayoutPolicy
    import slaythespire as sts
    model = _dummy_model()
    schema = dashboard.SlayTheSpireEnv(combat_action_contract=contract)
    model.observation_space = schema.observation_space
    schema.close()
    model.policy.small_playout = SmallPlayoutPolicy(sts.small_playout_key_count())
    session = _FixedModelOutputWatchSession(model, threading.Lock(), seed=0)
    try:
        session.configure_search(simulations=7)
        for _ in range(40):
            if session.info['phase'] == 'combat':
                break
            session.step()
        assert session.info['phase'] == 'combat'
        position = session.position
        assert session.decision['search']['stats']['simulations'] == 7
        before = session.env.unwrapped.thick.obs_canonical()
        session.configure_search(simulations=19)
        assert session.position == position
        assert session.env.unwrapped.thick.obs_canonical() == before
        assert session.decision['search']['stats']['simulations'] == 19
        bits = session.decision['native_action_bits']
        assert bits in session.env.unwrapped.native_combat_actions()
        session.step()
        session.undo()
        assert session.position == position
        assert session.decision['native_action_bits'] == bits
        session.configure_search(simulations=0)
        assert session.decision['source'] == 'compact_policy'
        assert session.decision['native_action_bits'] in session.env.unwrapped.native_combat_actions()
    finally:
        session.close()


def test_native_skip_battles_resolves_first_monster_without_combat():
    env = SimGameEnv(global_map_view=True)
    env.reset(seed=0)
    for _ in range(16):
        if env.phase() == "map":
            break
        legal = env.legal_semantic_actions()
        assert legal
        env.step(legal[0])
    assert env.phase() == "map"

    hp = int(env.gc.cur_hp)
    env.gc.skip_battles = True
    map_action = next(
        action for action in env.legal_semantic_actions()
        if action[0] == ga.MAP
    )
    env.step(map_action)

    assert env.bc is None
    assert env.phase() == "combat_reward"
    assert env.gc.cur_hp == hp


def test_watch_route_fast_forward_stops_at_real_boss_and_replays():
    session = _FirstLegalWatchSession(_dummy_model(), threading.Lock(), seed=0)
    try:
        snapshot = session.reset(0, simulations=0, target_boss_act=1)
        thick = session.env.unwrapped.thick

        assert snapshot["phase"] == "combat"
        assert snapshot["upper_strategy"] == "rules"
        assert snapshot["upper_strategy_version"] == "ironclad-overworld-v3"
        assert snapshot["route_preprocess"] == {
            "enabled": True,
            "target_act": 1,
            "reached": True,
            "boss_step": snapshot["step"],
            "baseline_hp": 80,
            "ignored_resources": ["hp", "max_hp", "potions"],
            "route_steps": snapshot["step"],
        }
        assert str(thick.gc.cur_room).split(".")[-1] == "BOSS"
        assert thick.gc.act == 1
        assert thick.gc.cur_hp == thick.gc.max_hp == 80
        assert set(thick.gc.overworld_potions) == {"EMPTY_POTION_SLOT"}
        assert session.route_step_modes[-1] == dashboard._ROUTE_STEP_ENTER_BOSS
        assert dashboard._ROUTE_STEP_NORMAL not in session.route_step_modes

        boss_name = str(thick.gc.encounter)
        boss_step = snapshot["step"]
        before_boss = session.undo()
        assert before_boss["phase"] == "map"
        assert before_boss["route_preprocess"]["reached"] is False
        assert session.env.unwrapped.thick.gc.cur_map_node_y == 14

        replayed = session.step()
        assert replayed["step"] == boss_step
        assert replayed["phase"] == "combat"
        assert replayed["route_preprocess"]["reached"] is True
        assert str(session.env.unwrapped.thick.gc.encounter) == boss_name
    finally:
        session.close()


def test_watch_defaults_to_shared_upper_rules_and_keeps_model_option(monkeypatch):
    session = _FixedModelOutputWatchSession(
        _dummy_model(), threading.Lock(), seed=10000
    )
    try:
        snapshot = session.snapshot()
        assert snapshot["upper_strategy"] == "rules"
        assert snapshot["combat_strategy"] == "model"
        assert snapshot["combat_strategy_version"] == "checkpoint-model+configured-search"
        assert snapshot["upper_fallback"] == "masked_argmax"
        assert snapshot["decision"]["source"] == "overworld_rules"
        assert snapshot["decision"]["action"] == session.env.overworld_rule_decision()

        snapshot = session.reset(10000, simulations=0, upper_strategy="model")
        assert snapshot["upper_strategy"] == "model"
        assert snapshot["decision"]["source"] == "model"
        assert snapshot["decision"]["action"] == snapshot["decision"]["model_action"]

        session.upper_strategy = "rules"
        monkeypatch.setattr(session.env, "overworld_rule_decision", lambda: None)
        session._prepare_decision()
        assert session.decision["source"] == "model_fallback"
        assert session.decision["action"] == session.decision["model_action"]

        with pytest.raises(ValueError, match="upper strategy"):
            session.reset(10000, simulations=0, upper_strategy="unknown")
    finally:
        session.close()


def test_watch_can_execute_replay_and_switch_from_turn_planner():
    session = _FixedModelOutputWatchSession(
        _dummy_model(), threading.Lock(), seed=0
    )
    try:
        session._turn_planner = dashboard.TurnPlannerStrategy(
            dashboard.TurnPlannerConfig(
                worlds=2,
                node_budget=32,
                beam_width=8,
                max_depth=16,
                lookahead_turns=1,
                master_nonce=20260918,
            )
        )
        for _ in range(64):
            if session.info["phase"] == "combat":
                break
            session.step()
        assert session.info["phase"] == "combat"
        snapshot = session.configure_search(
            simulations=0, combat_strategy="planner"
        )

        assert snapshot["combat_strategy"] == "planner"
        assert snapshot["combat_strategy_version"] == dashboard.rulebase_turn_policy_version(session._turn_planner.config)
        assert snapshot["decision"]["source"] == "turn-planner"
        assert snapshot["decision"]["native_direct"] is True
        native = snapshot["decision"]["native_action_bits"]
        assert native in session.env.native_combat_actions()
        assert snapshot["decision"]["combat_planner"]["worlds_completed"] == 2

        combat_step = snapshot["step"]
        after = session.step()
        assert after["step"] == combat_step + 1
        assert session.native_history[combat_step] == native
        assert session.native_direct_history[combat_step] is True
        after_state = session.env.thick.bc.to_dict()

        before = session.undo()
        assert before["step"] == combat_step
        replayed = session.step()
        assert replayed["step"] == combat_step + 1
        assert session.env.thick.bc.to_dict() == after_state

        snapshot = session.configure_search(combat_strategy="model")
        assert snapshot["combat_strategy"] == "model"
        assert snapshot["decision"]["source"] == "model"

        with pytest.raises(ValueError, match="combat strategy"):
            session.configure_search(combat_strategy="unknown")
    finally:
        session.close()


def test_neural_watch_rulebase_multi_prefix_undo_replays_same_subset(replayable_selection_env, monkeypatch):
    from env.core import native_actions as C
    from env.search.turn_policy import RulebaseTurnPolicy
    env = replayable_selection_env
    session = _FixedModelOutputWatchSession(_dummy_model(), threading.Lock(), seed=42)
    session.env.close()
    session.env = env
    session._run_backend_factory = env._backend_factory
    original = RulebaseTurnPolicy.choose
    calls = []

    def choose(policy, current, **kwargs):
        if current._multi_desc() is not None:
            calls.append(kwargs['state_nonce'])
            return (3 << 29) | 0b101, {'worlds_completed': 2}
        return original(policy, current, **kwargs)

    monkeypatch.setattr(RulebaseTurnPolicy, 'choose', choose)
    try:
        session.manual_mode = True
        session.manual_action(env._native_action_to_thin(0), session.action_revision)
        session.manual_mode = False
        session.configure_search(simulations=0, paper_full_gumbel=False, combat_strategy='planner')
        assert session.decision['source'] == 'turn-planner'
        assert not session.decision['native_direct']
        assert session.decision['action'] == C.AUTOREGRESSIVE_MULTI
        before = env.thick.bc.to_dict()
        session.step()
        assert env.thick.bc.to_dict() == before  # First token must not exhaust cards yet.
        assert session.decision['action'] == C.AUTOREGRESSIVE_MULTI + 2
        first_token_step = session.position
        with pytest.raises(ValueError, match='完成当前多选'):
            session.configure_search(manual_mode=True)
        assert session.manual_mode is False
        session.step()
        assert session.decision['action'] == C.AUTOREGRESSIVE_STOP
        session.step()
        after = env.thick.bc.to_dict()
        assert len(after['exhaust']) == 3  # Purity itself, plus exactly two selected strikes.
        assert len(calls) == 1
        session.seek(first_token_step)
        assert session.decision['action'] == C.AUTOREGRESSIVE_MULTI + 2
        session.step()
        session.step()
        assert env.thick.bc.to_dict() == after
        assert len(calls) == 1  # Saved subset, not a new search with a different prefix.
    finally:
        session.close()
