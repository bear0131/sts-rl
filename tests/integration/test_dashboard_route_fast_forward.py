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
        assert snapshot["upper_strategy_version"] == "ironclad-overworld-v1"
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


def test_watch_can_switch_between_shared_combat_rules_and_model():
    session = _FixedModelOutputWatchSession(
        _dummy_model(), threading.Lock(), seed=0
    )
    state = {
        "encounter": "Cultist",
        "player": {
            "hp": 80,
            "max_hp": 80,
            "energy": 3,
            "block": 0,
            "powers": [],
            "counters": {},
        },
        "monsters": [
            {
                "hp": 48,
                "max_hp": 48,
                "block": 0,
                "move": "INCANTATION",
                "powers": [],
            }
        ],
        "hand": [["STRIKE_RED", 0], ["DEFEND_RED", 0], ["BASH", 0]],
        "draw_pile": [],
        "discard_pile": [],
        "exhaust_pile": [],
        "relics": [],
    }
    try:
        session.configure_search(simulations=0, combat_strategy="rules")
        snapshot = session.apply_custom_state(state)

        assert snapshot["combat_strategy"] == "rules"
        assert snapshot["combat_strategy_version"] == "ironclad-v3"
        assert snapshot["decision"]["source"] == "rules-ironclad-v3"
        assert snapshot["decision"]["combat_rule"]["rule_decisions"] == 1
        assert snapshot["decision"]["action"] in np.flatnonzero(
            session.env.action_masks()
        )
        assert snapshot["decision"]["action"] in {
            item["action"] for item in snapshot["decision"]["top_actions"]
        }

        snapshot = session.configure_search(combat_strategy="model")
        assert snapshot["combat_strategy"] == "model"
        assert snapshot["decision"]["source"] == "model"

        with pytest.raises(ValueError, match="combat strategy"):
            session.configure_search(combat_strategy="unknown")
    finally:
        session.close()
