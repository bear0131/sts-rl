"""Local browser workbench for inspecting and evaluating sts-rl checkpoints."""

from __future__ import annotations

import argparse
import json
import math
import mimetypes
import os
import re
import subprocess
import sys
import threading
import time
import uuid
import zipfile
from collections import Counter
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np
from sb3_contrib import MaskablePPO

from env.core import action_space as A
from env.core.run_env import SlayTheSpireEnv
from rl.eval import is_win
from rl.visualize_card_vectors import (
    extract_state_dict_vectors,
    extract_vectors,
    pca_project,
    resolve_encoder,
)
from rl.watch import _action_dist
from env.core.cards_vocab import CARD_INDEX
from env.core.card_language import (
    TOKEN_DOC as CARD_LANGUAGE_DOC,
    TOKEN_VOCAB as CARD_LANGUAGE_TOKENS,
)
from env.core.combat_language import TOKEN_VOCAB as COMBAT_LANGUAGE_TOKENS
from env.core.event_descriptions import describe_event
from env.core.ironclad_cards import IRONCLAD_CARDS, card_numbers
from env.core.card_language import STATUS_CARDS
from rl.rllib_inference import RLLibInferenceModel

# Register custom policy classes before checkpoint loading.
from . import model as _model  # noqa: F401
from . import pointer_policy as _pointer_policy  # noqa: F401


STATIC_DIR = Path(__file__).with_name("dashboard_static")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
UPLOAD_DIR = Path("rl/runs/dashboard/uploads")
HYBRID_EVAL_ROOT = PROJECT_ROOT / "rl/runs/eval-hybrid"
DEFAULT_HYBRID_POLICY_STATE = Path(
    os.environ.get(
        "STS_RL_POLICY_STATE",
        r"C:\Users\Tony2\Downloads\policy_state.pkl",
    )
)
DEFAULT_HYBRID_COMBAT_CHECKPOINT = Path(
    os.environ.get(
        "STS_RL_COMBAT_CHECKPOINT",
        r"C:\Users\Tony2\Downloads\maskable_ppo_6400000_steps.zip",
    )
)
MAX_UPLOAD_BYTES = 512 * 1024 * 1024

IRONCLAD_VECTOR_CARDS = list(IRONCLAD_CARDS)


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _clamp_int(value, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return max(minimum, min(parsed, maximum))


def _file_info(path: Path) -> dict:
    if not path.is_file():
        return {"path": str(path), "exists": False}
    stat = path.stat()
    return {
        "path": str(path),
        "exists": True,
        "size": stat.st_size,
        "mtime": stat.st_mtime,
    }


def _tail_text(path: Path, max_lines: int = 80) -> str:
    if not path.is_file():
        return ""
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ""
    return "\n".join(lines[-max_lines:])


def _parse_hybrid_progress(stdout_tail: str) -> dict:
    condition_progress: dict[str, dict] = {}
    loaded: dict[str, int] = {}
    current = None
    progress_re = re.compile(
        r"\[hybrid-eval\]\s+"
        r"([A-Za-z0-9_]+)\s+"
        r"(\d+)/(\d+)\s+floor_mean=([0-9.]+)"
    )
    loaded_re = re.compile(r"loaded RLlib steps=(\d+) and SB3 steps=(\d+)")
    for line in stdout_tail.splitlines():
        match = progress_re.search(line)
        if match:
            condition, done, episodes, floor_mean = match.groups()
            current = condition
            condition_progress[condition] = {
                "completed": int(done),
                "episodes": int(episodes),
                "floor_mean": float(floor_mean),
            }
            continue
        match = loaded_re.search(line)
        if match:
            loaded = {
                "rllib_steps": int(match.group(1)),
                "sb3_steps": int(match.group(2)),
            }
    return {
        "conditions": condition_progress,
        "current_condition": current,
        "loaded": loaded,
    }


class WatchSession:
    def __init__(self, model: Any, model_lock: threading.Lock, seed: int = 0):
        self.model = model
        self.model_lock = model_lock
        self.env = SlayTheSpireEnv(
            global_map_view=_model.uses_global_map_observation(model),
            card_language=_model.uses_card_language(model),
            combat_language=_model.uses_combat_language(model),
            deck_size_observation=_model.uses_deck_size_observation(model),
            neow_observation=_model.uses_neow_observation(model),
            event_option_observation=_model.uses_event_option_observation(model),
        )
        self.seed = seed
        self.history: list[int] = []
        self.position = 0
        self.timeline: list[dict] = []
        self.timeline_exhausted = False
        self.obs = None
        self.info = None
        self.terminated = False
        self.truncated = False
        self.total_reward = 0.0
        self.last_action = None
        self.last_reward = 0.0
        self.decision = None
        self.map_state: dict | None = None
        self.map_states: dict[int, dict] = {}
        self.card_reward_choices: list[dict] = []
        self.reset(seed)

    def close(self):
        self.env.close()

    def reset(self, seed: int, precompute: bool = False):
        self.seed = int(seed)
        self.history = []
        self.position = 0
        self.timeline = []
        self.timeline_exhausted = False
        self.obs, self.info = self.env.reset(seed=self.seed)
        self.terminated = self.truncated = False
        self.total_reward = 0.0
        self.last_action = None
        self.last_reward = 0.0
        self.map_state = None
        self.map_states = {}
        self.card_reward_choices = []
        self._prepare_decision()
        if precompute:
            return self.precompute_run()
        return self.snapshot()

    def precompute_run(self):
        original_position = self.position
        steps = 0
        while not (self.terminated or self.truncated):
            previous_position = self.position
            self._advance_one()
            steps += 1
            if self.position == previous_position:
                break
            if steps >= 5000:
                raise RuntimeError("route preprocessing exceeded 5000 actions")
        if self.terminated or self.truncated:
            self.timeline_exhausted = True
        return self.seek(original_position)

    def advance_to_timeline(self, target_index: int):
        target_index = int(target_index)
        if target_index < 0 or target_index >= 48:
            raise ValueError(f"timeline index must be between 0 and 47, got {target_index}")
        if target_index < len(self.timeline):
            return self.seek(self.timeline[target_index]["step"])

        steps = 0
        while len(self.timeline) <= target_index and not (
            self.terminated or self.truncated
        ):
            previous_position = self.position
            self._advance_one()
            steps += 1
            if self.position == previous_position:
                break
            if steps >= 5000:
                raise RuntimeError("fast-forward exceeded 5000 actions")

        if self.terminated or self.truncated:
            self.timeline_exhausted = True
        if target_index < len(self.timeline):
            return self.snapshot()
        return self.snapshot()

    def preview_map(self, act: int):
        act = int(act)
        if act < 1 or act > 3:
            raise ValueError(f"act must be between 1 and 3, got {act}")
        original_position = self.position
        # Each act contributes 15 visible map rooms plus a y=15 boss entry.
        # Preview through the last visible room; entering a later act naturally
        # records the previous act's boss checkpoint as well.
        target_index = (act - 1) * 16 + 14
        if target_index >= len(self.timeline) and not self.timeline_exhausted:
            self.advance_to_timeline(target_index)
        if self.position != original_position:
            self.seek(original_position)
        return self.snapshot()

    def _prepare_decision(self):
        if self.terminated or self.truncated:
            self.decision = None
            return
        mask = self.env.action_masks()
        with self.model_lock:
            probs, value = _action_dist(self.model, self.obs, mask)
        legal = np.flatnonzero(mask)
        if probs is None:
            chosen = int(legal[0])
            ranked = [(chosen, None)]
        else:
            chosen = int(max(legal, key=lambda i: probs[i]))
            ranked = sorted(legal, key=lambda i: probs[i], reverse=True)[:6]
            ranked = [(int(i), float(probs[i])) for i in ranked]
        self.decision = {
            "action": chosen,
            "label": self.env.describe_action(chosen),
            "value": value,
            "animation": self._action_animation(chosen),
            "top_actions": [
                {"action": aid, "label": self.env.describe_action(aid), "prob": prob}
                for aid, prob in ranked
            ],
        }

    def _action_animation(self, action: int):
        play_base = A.seg_base("PLAY_CARD")
        target_base = A.seg_base("PICK_TARGET")
        if play_base <= action < play_base + A.HAND_SLOTS:
            return {"kind": "play_card", "slot": action - play_base}
        if target_base <= action < target_base + A.MAX_MONSTERS:
            return {"kind": "target", "slot": action - target_base}
        if action == A.seg_base("END_TURN"):
            return {"kind": "end_turn"}
        return None

    def step(self):
        if self.decision is None:
            return self.snapshot()
        self._advance_one()
        return self.snapshot()

    def _advance_one(self):
        if self.decision is None:
            return
        if self.position < len(self.history):
            action = self.history[self.position]
        else:
            action = int(self.decision["action"])
            self.history.append(action)
        label = self.env.describe_action(action)
        animation = self._action_animation(action)
        canon_before = self.env.unwrapped.thick.obs_canonical()
        self._record_card_reward_choice(action, canon_before)
        self.obs, reward, self.terminated, self.truncated, self.info = self.env.step(action)
        self.position += 1
        if self.terminated or self.truncated:
            self.timeline_exhausted = True
        self.last_action = {
            "action": action,
            "label": label,
            "animation": animation,
        }
        self.last_reward = float(reward)
        self.total_reward += float(reward)
        self._prepare_decision()
        canon = self.env.unwrapped.thick.obs_canonical()
        map_snapshot = self._update_map(canon)
        self._record_timeline(map_snapshot, canon)

    def undo(self):
        if self.position <= 0:
            return self.snapshot()
        return self.seek(self.position - 1)

    def seek(self, target_step: int):
        target_step = int(target_step)
        if target_step < 0 or target_step > len(self.history):
            raise ValueError(
                f"step must be between 0 and {len(self.history)}, got {target_step}"
            )
        self.obs, self.info = self.env.reset(seed=self.seed)
        self.terminated = self.truncated = False
        self.total_reward = 0.0
        self.last_action = None
        self.last_reward = 0.0
        self.position = 0
        self.map_state = None
        self.card_reward_choices = []
        for action in self.history[:target_step]:
            label = self.env.describe_action(action)
            canon_before = self.env.unwrapped.thick.obs_canonical()
            self._record_card_reward_choice(
                action, canon_before, include_current_prob=False
            )
            self.obs, reward, self.terminated, self.truncated, self.info = self.env.step(action)
            self.position += 1
            self.total_reward += float(reward)
            self.last_reward = float(reward)
            self.last_action = {
                "action": action,
                "label": label,
                "animation": self._action_animation(action),
            }
            canon = self.env.unwrapped.thick.obs_canonical()
            map_snapshot = self._update_map(canon)
            self._record_timeline(map_snapshot, canon)
            if self.terminated or self.truncated:
                break
        self._prepare_decision()
        return self.snapshot()

    def _record_timeline(self, map_snapshot, canon):
        if not map_snapshot or not map_snapshot.get("current"):
            return
        act = int(map_snapshot["act"])
        x, y = (int(v) for v in map_snapshot["current"])
        if any(
            node["act"] == act and node["x"] == x and node["y"] == y
            for node in self.timeline
        ):
            return
        room = next(
            (
                node.get("room", "")
                for node in map_snapshot.get("nodes", [])
                if int(node.get("x", -1)) == x and int(node.get("y", -1)) == y
            ),
            "",
        )
        global_state = canon.get("global") or {}
        stats = self.info.get("run_stats", {}) if self.info else {}
        self.timeline.append({
            "step": self.position,
            "act": act,
            "floor": int(stats.get("floor", global_state.get("floor", 0)) or 0),
            "x": x,
            "y": y,
            "room": room,
        })

    def _update_map(self, canon):
        global_state = canon.get("global") or {}
        act = int(global_state.get("act", 0) or 0)
        if act <= 0:
            return None
        if act not in self.map_states:
            self.map_states[act] = {
                "act": act,
                "boss": global_state.get("boss", ""),
                "nodes": [],
                "visited": set(),
                "current": None,
            }

        state = self.map_states[act]
        self.map_state = state
        state["boss"] = global_state.get("boss", state["boss"])
        full_map = (canon.get("options") or {}).get("full_map")
        thick = self.env.unwrapped.thick
        gc = getattr(thick, "gc", None)
        current_x = getattr(gc, "cur_map_node_x", -1)
        current_y = getattr(gc, "cur_map_node_y", -1)
        if not full_map:
            run_map = getattr(thick, "rmap", None)
            if run_map is not None:
                full_map = run_map.full_view(current_x, current_y)
        if full_map:
            state["nodes"] = [dict(node) for node in full_map]
        if current_x >= 0 and current_y >= 0:
            current = (int(current_x), int(current_y))
            state["current"] = current
            state["visited"].add(current)
        for node in state["nodes"]:
            if node.get("current"):
                current = (int(node["x"]), int(node["y"]))
                state["current"] = current
                state["visited"].add(current)
        return self._map_snapshot(state)

    @staticmethod
    def _map_snapshot(state):
        return {
            "act": state["act"],
            "boss": state["boss"],
            "nodes": state["nodes"],
            "visited": [list(node) for node in sorted(state["visited"])],
            "current": list(state["current"]) if state["current"] else None,
        }

    def _record_card_reward_choice(
        self, action: int, canon: dict, *, include_current_prob: bool = True
    ) -> None:
        phase = self.info.get("phase") if self.info else canon.get("phase")
        if phase != "combat_reward":
            return
        options = canon.get("options") or {}
        cards = options.get("cards") or []
        if not cards:
            return
        card_base = A.seg_base("REWARD_CARD")
        skip_action = A.seg_base("REWARD_SKIP")
        selected_index = None
        skipped = False
        if card_base <= action < card_base + A.REWARD_CARD:
            selected_index = int(action - card_base)
            if selected_index >= len(cards):
                return
        elif action == skip_action:
            skipped = True
        else:
            return

        stats = self.info.get("run_stats", {}) if self.info else {}
        global_state = canon.get("global") or {}
        prob = None
        if include_current_prob and self.decision:
            for ranked in self.decision.get("top_actions", []):
                if int(ranked.get("action", -1)) == int(action):
                    prob = ranked.get("prob")
                    break
        self.card_reward_choices.append({
            "step": self.position,
            "floor": int(stats.get("floor", global_state.get("floor", 0)) or 0),
            "cards": list(cards),
            "selected_index": selected_index,
            "skipped": skipped,
            "relics": list(global_state.get("relics", [])),
            "action": int(action),
            "action_label": self.env.describe_action(action),
            "prob": prob,
        })

    def snapshot(self):
        canon = self.env.unwrapped.thick.obs_canonical()
        map_snapshot = self._update_map(canon)
        self._record_timeline(map_snapshot, canon)
        stats = self.info.get("run_stats", {}) if self.info else {}
        pending = self.env._pending
        pending_card_slot = (
            int(pending[1])
            if pending is not None and pending[0] == "play"
            else None
        )
        event = describe_event(
            (canon.get("options") or {}).get("event"),
            (canon.get("options") or {}).get("event_options", []),
        )
        if event is not None:
            event_base = A.seg_base("EVENT_OPT")
            for option in event["options"]:
                option["action"] = event_base + option["index"]
        return _jsonable({
            "seed": self.seed,
            "step": self.position,
            "max_step": len(self.history),
            "precomputed_steps": len(self.history) if self.timeline_exhausted else 0,
            "can_undo": self.position > 0,
            "done": self.terminated or self.truncated,
            "terminated": self.terminated,
            "truncated": self.truncated,
            "won": is_win(self.env) if self.terminated else False,
            "total_reward": self.total_reward,
            "last_reward": self.last_reward,
            "last_action": self.last_action,
            "decision": self.decision,
            "phase": self.info.get("phase") if self.info else None,
            "pending_card_slot": pending_card_slot,
            "stats": stats,
            "canon": canon,
            "event": event,
            "map": map_snapshot,
            "maps": [
                self._map_snapshot(self.map_states[act])
                for act in sorted(self.map_states)
            ],
            "timeline": self.timeline,
            "timeline_exhausted": self.timeline_exhausted,
            "card_reward_choices": self.card_reward_choices,
        })


class DashboardState:
    def __init__(self):
        self.lock = threading.RLock()
        self.model_lock = threading.Lock()
        self.checkpoint: Path | None = None
        self.checkpoint_format: str | None = None
        self.model: Any | None = None
        self.embedding: dict | None = None
        self.token_embeddings: dict = {}
        self.watch: WatchSession | None = None
        self.eval_status: dict = {"state": "idle"}
        self.eval_token = ""
        self.hybrid_eval_status: dict = {
            "state": "idle",
            "defaults": self.hybrid_eval_defaults(),
        }
        self.hybrid_eval_token = ""
        self.hybrid_eval_process: subprocess.Popen | None = None
        self.error: str | None = None

    @staticmethod
    def hybrid_eval_defaults() -> dict:
        return {
            "policy_state": str(DEFAULT_HYBRID_POLICY_STATE),
            "combat_checkpoint": str(DEFAULT_HYBRID_COMBAT_CHECKPOINT),
            "route_phase": "combat",
            "episodes": 100,
            "seed": 10_000,
            "ascension": 0,
            "num_envs": 16,
        }

    def public_state(self):
        with self.lock:
            return {
                "checkpoint": self.checkpoint.name if self.checkpoint else None,
                "checkpoint_path": str(self.checkpoint) if self.checkpoint else None,
                "checkpoint_format": self.checkpoint_format,
                "capabilities": {
                    "global_map": _model.uses_global_map_observation(self.model)
                    if self.model else False,
                    "map_dag": _model.uses_map_dag(self.model) if self.model else False,
                    "card_language": _model.uses_card_language(self.model)
                    if self.model else False,
                    "combat_language": _model.uses_combat_language(self.model)
                    if self.model else False,
                    "deck_size": _model.uses_deck_size_observation(self.model)
                    if self.model else False,
                    "neow": _model.uses_neow_observation(self.model)
                    if self.model else False,
                    "event_option": _model.uses_event_option_observation(self.model)
                    if self.model else False,
                },
                "embedding": self.embedding,
                "token_embeddings": self.token_embeddings,
                "card_catalog": {
                    card_id: {
                        "name_zh": info.name_zh,
                        "kind": info.kind,
                        "cost": list(info.cost),
                        "damage": list(info.damage),
                        "block": list(info.block),
                        "effect_en": info.effect_en,
                        "effect_zh": info.effect_zh,
                    }
                    for card_id, info in IRONCLAD_CARDS.items()
                } | {
                    card_id: {
                        "name_zh": info["name_zh"],
                        "kind": "status",
                        "cost": [-1, -1],
                        "damage": [0, 0],
                        "block": [0, 0],
                        "effect_en": info.get("effect_en", ""),
                        "effect_zh": info.get("effect_zh", ""),
                    }
                    for card_id, info in STATUS_CARDS.items()
                },
                "watch": self.watch.snapshot() if self.watch else None,
                "evaluation": self.eval_status,
                "hybrid_evaluation": self._hybrid_eval_public_locked(),
                "error": self.error,
            }

    def _hybrid_eval_public_locked(self) -> dict:
        status = dict(self.hybrid_eval_status)
        status.setdefault("defaults", self.hybrid_eval_defaults())
        process = self.hybrid_eval_process
        if status.get("state") == "running" and process is not None:
            return_code = process.poll()
            if return_code is not None:
                status["return_code"] = return_code
        return status

    def load_checkpoint(self, path: Path, episodes: int):
        suffix = path.suffix.lower()
        if suffix == ".zip":
            model = MaskablePPO.load(path, device="auto")
            checkpoint_format = "sb3"
        elif suffix == ".pkl":
            model = RLLibInferenceModel(path, device="auto")
            checkpoint_format = "rllib"
        else:
            raise ValueError(
                "checkpoint must be an SB3 .zip or RLlib policy_state.pkl"
            )
        model.set_random_seed(0)
        embedding = self._embedding(model)
        token_embeddings = self._token_embeddings(model)
        watch = WatchSession(model, self.model_lock, seed=0)
        with self.lock:
            old_watch = self.watch
            self.checkpoint = path
            self.checkpoint_format = checkpoint_format
            self.model = model
            self.embedding = embedding
            self.token_embeddings = token_embeddings
            self.watch = watch
            self.error = None
        if old_watch:
            old_watch.close()
        self.start_evaluation(episodes=episodes, seed=0)

    def _embedding(self, model: Any):
        try:
            encoder = resolve_encoder(model)
            base, description = extract_vectors(encoder, "hand", upgrades=0)
            upgraded, _ = extract_vectors(encoder, "hand", upgrades=1)
        except RuntimeError:
            # Older flat policies have no pointer core; retain their state-dict
            # compatibility path (they also predate the card language).
            state = model.policy.state_dict()
            base, description = extract_state_dict_vectors(
                state, "hand", upgrades=0
            )
            upgraded, _ = extract_state_dict_vectors(
                state, "hand", upgrades=1
            )
        indices = np.asarray([CARD_INDEX[name] for name in IRONCLAD_VECTOR_CARDS])
        vectors = np.concatenate([base[indices], upgraded[indices]], axis=0)
        coords, ratio, norms = pca_project(vectors, dim=2, normalize=False)
        points = []
        for variant in (0, 1):
            for i, card_id in enumerate(IRONCLAD_VECTOR_CARDS):
                row = variant * len(IRONCLAD_VECTOR_CARDS) + i
                cost, damage, block = card_numbers(card_id, variant)
                info = IRONCLAD_CARDS[card_id]
                points.append({
                    "card_id": card_id,
                    "name": info.name_zh + ("+" if variant else ""),
                    "kind": info.kind,
                    "upgrade": variant,
                    "x": float(coords[row, 0]),
                    "y": float(coords[row, 1]),
                    "norm": float(norms[row]),
                    "cost": cost,
                    "damage": damage,
                    "block": block,
                })
        return {
            "points": points,
            "explained_ratio": ratio.tolist(),
            "description": description,
            "card_count": len(IRONCLAD_VECTOR_CARDS),
        }

    @staticmethod
    def _token_group(token: str) -> str:
        if token == "PAD" or token.endswith(("_BEGIN", "_END")):
            return "structure"
        if token.startswith(("DIGIT_", "NUM_", "VALUE_")):
            return "number"
        if token.startswith(("TYPE_", "RARITY_")) or token == "COST":
            return "metadata"
        if token in {"DO", "TRIGGER", "IF", "TRAIT"}:
            return "operator"
        return "semantic"

    def _token_embeddings(self, model: Any):
        """Project each learned DSL token table in its own parameter space."""
        try:
            encoder = resolve_encoder(model)
        except RuntimeError:
            return {}

        sources = []
        card = getattr(getattr(encoder, "shared", {}), "card", None)
        card_tokens = getattr(card, "token_embedding", None)
        if card_tokens is not None:
            sources.append((
                "card", "卡牌语言", CARD_LANGUAGE_TOKENS,
                CARD_LANGUAGE_DOC, card_tokens.weight,
            ))
        combat = getattr(encoder, "combat_language", None)
        combat_tokens = getattr(combat, "token_embedding", None)
        if combat_tokens is not None:
            sources.append((
                "combat", "战斗语言", COMBAT_LANGUAGE_TOKENS,
                {}, combat_tokens.weight,
            ))

        result = {}
        for source, label, vocabulary, docs, weight in sources:
            vectors = weight.detach().float().cpu().numpy()
            if vectors.shape[0] != len(vocabulary):
                continue
            coords, ratio, norms = pca_project(vectors, dim=2, normalize=False)
            points = []
            for index, token in enumerate(vocabulary):
                doc = docs.get(token)
                points.append({
                    "index": index,
                    "token": token,
                    "group": self._token_group(token),
                    "x": float(coords[index, 0]),
                    "y": float(coords[index, 1]),
                    "norm": float(norms[index]),
                    "description_en": doc[0] if doc else None,
                    "description_zh": doc[1] if doc else None,
                })
            result[source] = {
                "label": label,
                "points": points,
                "explained_ratio": ratio.tolist(),
            }
        return result

    def start_evaluation(self, episodes: int, seed: int):
        episodes = max(1, min(int(episodes), 10_000))
        token = uuid.uuid4().hex
        with self.lock:
            if self.model is None:
                raise RuntimeError("no checkpoint loaded")
            model = self.model
            self.eval_token = token
            self.eval_status = {
                "state": "running", "episodes": episodes, "completed": 0,
                "seed": seed, "started_at": time.time(), "death_floors": {},
                "death_seeds": {}, "win_seeds": [], "current_episode": 1,
                "current_steps": 0, "current_floor": 0,
                "current_phase": None, "current_action": None,
            }
        threading.Thread(
            target=self._evaluate,
            args=(token, model, episodes, int(seed)),
            daemon=True,
            name="dashboard-eval",
        ).start()

    def _evaluate(self, token: str, model: Any, episodes: int, seed: int):
        env = SlayTheSpireEnv(
            global_map_view=_model.uses_global_map_observation(model),
            card_language=_model.uses_card_language(model),
            combat_language=_model.uses_combat_language(model),
            deck_size_observation=_model.uses_deck_size_observation(model),
            neow_observation=_model.uses_neow_observation(model),
            event_option_observation=_model.uses_event_option_observation(model),
        )
        death_floors: Counter[int] = Counter()
        death_seeds: dict[int, list[int]] = {}
        win_seeds: list[int] = []
        result_counts: Counter[str] = Counter()
        floors: list[int] = []
        lengths: list[int] = []
        try:
            for episode in range(episodes):
                obs, info = env.reset(seed=seed + episode)
                terminated = truncated = False
                steps = 0
                while not (terminated or truncated):
                    with self.lock:
                        if token != self.eval_token:
                            return
                    mask = env.action_masks()
                    phase_before = info.get("phase")
                    with self.model_lock:
                        action, _ = model.predict(obs, action_masks=mask, deterministic=True)
                    action_id = int(np.asarray(action).item())
                    action_label = env.describe_action(action_id)
                    obs, _reward, terminated, truncated, info = env.step(
                        action_id
                    )
                    steps += 1
                    if steps == 1 or steps % 25 == 0:
                        live_stats = info.get("run_stats", {})
                        with self.lock:
                            if token != self.eval_token:
                                return
                            self.eval_status.update({
                                "current_episode": episode + 1,
                                "current_steps": steps,
                                "current_floor": int(live_stats.get("floor", 0)),
                                "current_phase": info.get("phase"),
                                "current_phase_before": phase_before,
                                "current_action": action_id,
                                "current_action_label": action_label,
                                "current_action_legal": bool(mask[action_id]),
                                "current_legal_actions": np.flatnonzero(mask).tolist(),
                                "last_progress_at": time.time(),
                            })
                stats = env.unwrapped.thick.run_stats()
                floor = int(stats["floor"])
                if is_win(env):
                    result = "win"
                    win_seeds.append(seed + episode)
                elif truncated and info.get("dead_screen"):
                    result = "dead_screen"
                elif truncated:
                    result = "other_truncation"
                else:
                    result = "death"
                    death_floors[floor] += 1
                    death_seeds.setdefault(floor, []).append(seed + episode)
                result_counts[result] += 1
                floors.append(floor)
                lengths.append(steps)
                elapsed = time.time() - self.eval_status["started_at"]
                done = episode + 1
                with self.lock:
                    if token != self.eval_token:
                        return
                    self.eval_status.update({
                        "completed": done,
                        "death_floors": dict(sorted(death_floors.items())),
                        "death_seeds": {
                            floor: seeds
                            for floor, seeds in sorted(death_seeds.items())
                        },
                        "win_seeds": win_seeds,
                        "results": dict(result_counts),
                        "floor_mean": float(np.mean(floors)),
                        "floor_median": float(np.median(floors)),
                        "steps_mean": float(np.mean(lengths)),
                        "elapsed_seconds": elapsed,
                        "eta_seconds": elapsed / done * (episodes - done),
                    })
            with self.lock:
                if token == self.eval_token:
                    self.eval_status["state"] = "complete"
        except Exception as exc:
            with self.lock:
                if token == self.eval_token:
                    self.eval_status.update({"state": "error", "error": str(exc)})
        finally:
            env.close()

    @staticmethod
    def _request_path(value, default: Path) -> Path:
        path = Path(value or default)
        return path if path.is_absolute() else PROJECT_ROOT / path

    def start_hybrid_evaluation(self, config: dict):
        policy_state = self._request_path(
            config.get("policy_state"),
            DEFAULT_HYBRID_POLICY_STATE,
        )
        combat_checkpoint = self._request_path(
            config.get("combat_checkpoint"),
            DEFAULT_HYBRID_COMBAT_CHECKPOINT,
        )
        if not policy_state.is_file():
            raise FileNotFoundError(f"policy_state not found: {policy_state}")
        if not combat_checkpoint.is_file():
            raise FileNotFoundError(
                f"combat checkpoint not found: {combat_checkpoint}"
            )
        route_phase = str(config.get("route_phase", "combat")).strip() or "combat"
        if route_phase not in {"combat", "map"}:
            raise ValueError(f"unsupported route phase: {route_phase}")
        hybrid_label = f"hybrid_{route_phase}_sb3"

        episodes = _clamp_int(config.get("episodes"), 100, 1, 10_000)
        seed = _clamp_int(config.get("seed"), 10_000, -2_147_483_648, 2_147_483_647)
        ascension = _clamp_int(config.get("ascension"), 0, 0, 20)
        num_envs = _clamp_int(config.get("num_envs"), 16, 1, 256)
        progress_every = _clamp_int(config.get("progress_every"), 10, 0, 10_000)
        heartbeat_seconds = _clamp_int(
            config.get("heartbeat_seconds"), 30, 0, 3_600
        )
        token = uuid.uuid4().hex

        requested_output = config.get("output_dir")
        if requested_output:
            output_dir = self._request_path(requested_output, HYBRID_EVAL_ROOT)
        else:
            stamp = time.strftime("%Y%m%d-%H%M%S")
            output_dir = HYBRID_EVAL_ROOT / f"dashboard-{stamp}-{token[:8]}"
        output_dir.mkdir(parents=True, exist_ok=True)
        stdout_path = output_dir / "stdout.log"
        stderr_path = output_dir / "stderr.log"
        summary_path = output_dir / "summary.json"
        command = [
            sys.executable,
            "-u",
            "-m",
            "rl.eval_hybrid",
            str(policy_state),
            "--combat-checkpoint",
            str(combat_checkpoint),
            "--episodes",
            str(episodes),
            "--seed",
            str(seed),
            "--route-phase",
            route_phase,
            "--ascension",
            str(ascension),
            "--num-envs",
            str(num_envs),
            "--progress-every",
            str(progress_every),
            "--heartbeat-seconds",
            str(heartbeat_seconds),
            "--output-dir",
            str(output_dir),
        ]

        creationflags = (
            subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        )
        with self.lock:
            self._terminate_hybrid_eval_locked("replaced")
            self.hybrid_eval_token = token
            self.hybrid_eval_status = {
                "state": "starting",
                "token": token,
                "started_at": time.time(),
                "episodes": episodes,
                "seed": seed,
                "route_phase": route_phase,
                "hybrid_label": hybrid_label,
                "ascension": ascension,
                "num_envs": num_envs,
                "completed": 0,
                "total": episodes * 2,
                "policy_state": str(policy_state),
                "combat_checkpoint": str(combat_checkpoint),
                "policy_state_file": _file_info(policy_state),
                "combat_checkpoint_file": _file_info(combat_checkpoint),
                "output_dir": str(output_dir),
                "stdout_log": str(stdout_path),
                "stderr_log": str(stderr_path),
                "summary_path": str(summary_path),
                "command": command,
                "defaults": self.hybrid_eval_defaults(),
            }
        try:
            with stdout_path.open("w", encoding="utf-8") as stdout, (
                stderr_path.open("w", encoding="utf-8")
            ) as stderr:
                process = subprocess.Popen(
                    command,
                    cwd=PROJECT_ROOT,
                    stdout=stdout,
                    stderr=stderr,
                    creationflags=creationflags,
                )
        except Exception:
            with self.lock:
                if token == self.hybrid_eval_token:
                    self.hybrid_eval_status["state"] = "error"
            raise

        with self.lock:
            if token != self.hybrid_eval_token:
                if process.poll() is None:
                    process.terminate()
                return self._hybrid_eval_public_locked()
            self.hybrid_eval_process = process
            self.hybrid_eval_status["state"] = "running"

        threading.Thread(
            target=self._monitor_hybrid_evaluation,
            args=(token, process, output_dir, stdout_path, stderr_path),
            daemon=True,
            name="dashboard-hybrid-eval",
        ).start()
        with self.lock:
            return self._hybrid_eval_public_locked()

    def _terminate_hybrid_eval_locked(self, reason: str):
        process = self.hybrid_eval_process
        if process is None or process.poll() is not None:
            return
        self.hybrid_eval_status.update({"state": "stopping", "stop_reason": reason})
        process.terminate()
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            process.kill()

    def stop_hybrid_evaluation(self):
        with self.lock:
            self._terminate_hybrid_eval_locked("stopped")
            if self.hybrid_eval_status.get("state") == "stopping":
                self.hybrid_eval_status["state"] = "stopped"
            return self._hybrid_eval_public_locked()

    def _update_hybrid_eval_progress(
        self,
        token: str,
        stdout_path: Path,
        stderr_path: Path,
    ) -> bool:
        stdout_tail = _tail_text(stdout_path)
        stderr_tail = _tail_text(stderr_path, max_lines=40)
        parsed = _parse_hybrid_progress(stdout_tail)
        conditions = parsed["conditions"]
        completed = sum(
            int(item.get("completed", 0)) for item in conditions.values()
        )
        with self.lock:
            if token != self.hybrid_eval_token:
                return False
            total = int(
                self.hybrid_eval_status.get(
                    "total",
                    int(self.hybrid_eval_status.get("episodes", 0)) * 2,
                )
            )
            elapsed = time.time() - float(
                self.hybrid_eval_status.get("started_at", time.time())
            )
            update = {
                "stdout_tail": stdout_tail,
                "stderr_tail": stderr_tail,
                "progress": conditions,
                "current_condition": parsed["current_condition"],
                "completed": completed,
                "elapsed_seconds": elapsed,
            }
            if parsed["loaded"]:
                update["loaded"] = parsed["loaded"]
            if completed > 0 and total > completed:
                update["eta_seconds"] = elapsed / completed * (total - completed)
            elif total and completed >= total:
                update["eta_seconds"] = 0
            self.hybrid_eval_status.update(update)
            return True

    @staticmethod
    def _load_hybrid_summary(summary_path: Path) -> dict | None:
        if not summary_path.is_file():
            return None
        try:
            return json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

    def _monitor_hybrid_evaluation(
        self,
        token: str,
        process: subprocess.Popen,
        output_dir: Path,
        stdout_path: Path,
        stderr_path: Path,
    ):
        summary_path = output_dir / "summary.json"
        while process.poll() is None:
            if not self._update_hybrid_eval_progress(
                token, stdout_path, stderr_path
            ):
                return
            time.sleep(1)

        self._update_hybrid_eval_progress(token, stdout_path, stderr_path)
        return_code = process.returncode
        summary = self._load_hybrid_summary(summary_path)
        with self.lock:
            if token != self.hybrid_eval_token:
                return
            previous_state = self.hybrid_eval_status.get("state")
            self.hybrid_eval_status["return_code"] = return_code
            self.hybrid_eval_status["finished_at"] = time.time()
            if previous_state == "stopping":
                self.hybrid_eval_status["state"] = "stopped"
            elif return_code == 0 and summary is not None:
                self.hybrid_eval_status.update({
                    "state": "complete",
                    "summary": summary,
                    "completed": self.hybrid_eval_status.get("total", 0),
                    "eta_seconds": 0,
                })
            elif return_code == 0:
                self.hybrid_eval_status.update({
                    "state": "error",
                    "error": "summary.json was not produced",
                })
            else:
                error = self.hybrid_eval_status.get("stderr_tail") or (
                    f"process exited with code {return_code}"
                )
                self.hybrid_eval_status.update({
                    "state": "error",
                    "error": error,
                })

APP = DashboardState()


class Handler(BaseHTTPRequestHandler):
    server_version = "StSRLWorkbench/1.0"

    def log_message(self, fmt, *args):
        print(f"[dashboard] {self.address_string()} {fmt % args}")

    def _send_json(self, payload, status=HTTPStatus.OK):
        data = json.dumps(_jsonable(payload), ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _read_json(self):
        length = int(self.headers.get("Content-Length", "0"))
        return json.loads(self.rfile.read(length) or b"{}")

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/api/state":
            self._send_json(APP.public_state())
            return
        if path == "/api/hybrid-eval/status":
            self._send_json(APP.public_state()["hybrid_evaluation"])
            return
        relative = "index.html" if path == "/" else path.lstrip("/")
        target = (STATIC_DIR / relative).resolve()
        if STATIC_DIR.resolve() not in target.parents and target != STATIC_DIR.resolve():
            self.send_error(HTTPStatus.FORBIDDEN)
            return
        if not target.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        data = target.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", mimetypes.guess_type(target.name)[0] or "application/octet-stream")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        path = urlparse(self.path).path
        try:
            if path == "/api/upload":
                self._upload()
            elif path == "/api/watch/reset":
                body = self._read_json()
                with APP.lock:
                    if not APP.watch:
                        raise RuntimeError("no checkpoint loaded")
                    result = APP.watch.reset(
                        int(body.get("seed", 0)),
                        precompute=bool(body.get("precompute", False)),
                    )
                self._send_json(result)
            elif path == "/api/watch/step":
                with APP.lock:
                    if not APP.watch:
                        raise RuntimeError("no checkpoint loaded")
                    result = APP.watch.step()
                self._send_json(result)
            elif path == "/api/watch/undo":
                with APP.lock:
                    if not APP.watch:
                        raise RuntimeError("no checkpoint loaded")
                    result = APP.watch.undo()
                self._send_json(result)
            elif path == "/api/watch/seek":
                body = self._read_json()
                with APP.lock:
                    if not APP.watch:
                        raise RuntimeError("no checkpoint loaded")
                    result = APP.watch.seek(int(body.get("step", 0)))
                self._send_json(result)
            elif path == "/api/watch/advance":
                body = self._read_json()
                with APP.lock:
                    if not APP.watch:
                        raise RuntimeError("no checkpoint loaded")
                    result = APP.watch.advance_to_timeline(int(body.get("index", 0)))
                self._send_json(result)
            elif path == "/api/watch/map-preview":
                body = self._read_json()
                with APP.lock:
                    if not APP.watch:
                        raise RuntimeError("no checkpoint loaded")
                    result = APP.watch.preview_map(int(body.get("act", 1)))
                self._send_json(result)
            elif path == "/api/eval/start":
                body = self._read_json()
                APP.start_evaluation(body.get("episodes", 50), body.get("seed", 0))
                self._send_json(APP.public_state()["evaluation"])
            elif path == "/api/hybrid-eval/start":
                body = self._read_json()
                self._send_json(APP.start_hybrid_evaluation(body))
            elif path == "/api/hybrid-eval/stop":
                self._send_json(APP.stop_hybrid_evaluation())
            else:
                self.send_error(HTTPStatus.NOT_FOUND)
        except Exception as exc:
            self._send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)

    def _upload(self):
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0 or length > MAX_UPLOAD_BYTES:
            raise ValueError("upload must be a non-empty file smaller than 512 MB")
        original = self.headers.get("X-Filename", "checkpoint.zip")
        filename = re.sub(r"[^A-Za-z0-9_.-]+", "_", Path(original).name)
        suffix = Path(filename).suffix.lower()
        if suffix not in {".zip", ".pkl"}:
            raise ValueError(
                "only SB3 .zip or RLlib policy_state.pkl checkpoints are accepted"
            )
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        target = UPLOAD_DIR / f"{int(time.time())}-{uuid.uuid4().hex[:8]}-{filename}"
        partial = target.with_suffix(target.suffix + ".part")
        try:
            remaining = length
            with partial.open("wb") as stream:
                while remaining:
                    chunk = self.rfile.read(min(1024 * 1024, remaining))
                    if not chunk:
                        raise ValueError(
                            f"upload ended early: received {length - remaining} "
                            f"of {length} bytes"
                        )
                    stream.write(chunk)
                    remaining -= len(chunk)
            with partial.open("rb") as stream:
                signature = stream.read(4)
            if suffix == ".zip":
                if signature not in (
                    b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08"
                ):
                    raise ValueError(
                        "uploaded file does not have a ZIP signature; re-download "
                        "the checkpoint and drag the .zip file itself"
                    )
            elif signature[:1] != b"\x80":
                raise ValueError(
                    "uploaded .pkl is not a binary RLlib policy_state checkpoint"
                )
            os.replace(partial, target)
            if suffix == ".zip":
                with zipfile.ZipFile(target) as archive:
                    if archive.testzip() is not None:
                        raise ValueError("checkpoint ZIP failed integrity validation")
            episodes = int(self.headers.get("X-Eval-Episodes", "50"))
            APP.load_checkpoint(target, episodes)
        except Exception:
            partial.unlink(missing_ok=True)
            target.unlink(missing_ok=True)
            raise
        self._send_json(APP.public_state(), HTTPStatus.CREATED)


def main():
    parser = argparse.ArgumentParser(description="Run the local sts-rl test dashboard.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--eval-episodes", type=int, default=50)
    args = parser.parse_args()
    if args.checkpoint:
        APP.load_checkpoint(args.checkpoint, args.eval_episodes)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"[dashboard] http://{args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
