"""Local browser workbench for inspecting and evaluating sts-rl checkpoints."""

from __future__ import annotations

import argparse
import json
import math
import mimetypes
import os
import re
import threading
import time
import uuid
import zipfile
from collections import Counter
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
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
from env.core.ironclad_cards import IRONCLAD_CARDS, card_numbers
from env.core.card_language import STATUS_CARDS

# Register custom policy classes before checkpoint loading.
from . import model as _model  # noqa: F401
from . import pointer_policy as _pointer_policy  # noqa: F401


STATIC_DIR = Path(__file__).with_name("dashboard_static")
UPLOAD_DIR = Path("rl/runs/dashboard/uploads")
MAX_UPLOAD_BYTES = 512 * 1024 * 1024

COMMON_CARDS = [
    "Anger", "Armaments", "Body Slam", "Clash", "Cleave", "Clothesline",
    "Flex", "Havoc", "Headbutt", "Heavy Blade", "Iron Wave",
    "Perfected Strike", "Pommel Strike", "Shrug It Off", "Sword Boomerang",
    "Thunderclap", "True Grit", "Twin Strike", "Warcry", "Wild Strike",
]
COMMON_ZH = {
    "Anger": "愤怒", "Armaments": "武装", "Body Slam": "全身撞击",
    "Clash": "交锋", "Cleave": "顺劈斩", "Clothesline": "金刚臂",
    "Flex": "活动肌肉", "Havoc": "破灭", "Headbutt": "头槌",
    "Heavy Blade": "重刃", "Iron Wave": "铁斩波",
    "Perfected Strike": "完美打击", "Pommel Strike": "剑柄打击",
    "Shrug It Off": "耸肩无视", "Sword Boomerang": "飞剑回旋镖",
    "Thunderclap": "闪电霹雳", "True Grit": "坚毅",
    "Twin Strike": "双重打击", "Warcry": "战吼",
    "Wild Strike": "狂野打击",
}


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


class WatchSession:
    def __init__(self, model: MaskablePPO, model_lock: threading.Lock, seed: int = 0):
        self.model = model
        self.model_lock = model_lock
        self.env = SlayTheSpireEnv(
            global_map_view=_model.uses_global_map_observation(model),
            card_language=_model.uses_card_language(model),
            combat_language=_model.uses_combat_language(model),
        )
        self.seed = seed
        self.history: list[int] = []
        self.obs = None
        self.info = None
        self.terminated = False
        self.truncated = False
        self.total_reward = 0.0
        self.last_action = None
        self.last_reward = 0.0
        self.decision = None
        self.map_state: dict | None = None
        self.reset(seed)

    def close(self):
        self.env.close()

    def reset(self, seed: int):
        self.seed = int(seed)
        self.history = []
        self.obs, self.info = self.env.reset(seed=self.seed)
        self.terminated = self.truncated = False
        self.total_reward = 0.0
        self.last_action = None
        self.last_reward = 0.0
        self.map_state = None
        self._prepare_decision()
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
        action = int(self.decision["action"])
        label = self.decision["label"]
        animation = self.decision.get("animation")
        self.obs, reward, self.terminated, self.truncated, self.info = self.env.step(action)
        self.history.append(action)
        self.last_action = {
            "action": action,
            "label": label,
            "animation": animation,
        }
        self.last_reward = float(reward)
        self.total_reward += float(reward)
        self._prepare_decision()
        return self.snapshot()

    def undo(self):
        if not self.history:
            return self.snapshot()
        replay = self.history[:-1]
        self.obs, self.info = self.env.reset(seed=self.seed)
        self.terminated = self.truncated = False
        self.total_reward = 0.0
        self.last_action = None
        self.last_reward = 0.0
        self.history = []
        self.map_state = None
        for action in replay:
            self.obs, reward, self.terminated, self.truncated, self.info = self.env.step(action)
            self.history.append(action)
            self.total_reward += float(reward)
            self._update_map(self.env.unwrapped.thick.obs_canonical())
            if self.terminated or self.truncated:
                break
        self._prepare_decision()
        return self.snapshot()

    def _update_map(self, canon):
        global_state = canon.get("global") or {}
        act = int(global_state.get("act", 0) or 0)
        if act <= 0:
            return None
        if self.map_state is None or self.map_state["act"] != act:
            self.map_state = {
                "act": act,
                "boss": global_state.get("boss", ""),
                "nodes": [],
                "visited": set(),
                "current": None,
            }

        state = self.map_state
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
        return {
            "act": state["act"],
            "boss": state["boss"],
            "nodes": state["nodes"],
            "visited": [list(node) for node in sorted(state["visited"])],
            "current": list(state["current"]) if state["current"] else None,
        }

    def snapshot(self):
        canon = self.env.unwrapped.thick.obs_canonical()
        map_snapshot = self._update_map(canon)
        stats = self.info.get("run_stats", {}) if self.info else {}
        pending = self.env._pending
        pending_card_slot = (
            int(pending[1])
            if pending is not None and pending[0] == "play"
            else None
        )
        return _jsonable({
            "seed": self.seed,
            "step": len(self.history),
            "can_undo": bool(self.history),
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
            "map": map_snapshot,
        })


class DashboardState:
    def __init__(self):
        self.lock = threading.RLock()
        self.model_lock = threading.Lock()
        self.checkpoint: Path | None = None
        self.model: MaskablePPO | None = None
        self.embedding: dict | None = None
        self.token_embeddings: dict = {}
        self.watch: WatchSession | None = None
        self.eval_status: dict = {"state": "idle"}
        self.eval_token = ""
        self.error: str | None = None

    def public_state(self):
        with self.lock:
            return {
                "checkpoint": self.checkpoint.name if self.checkpoint else None,
                "checkpoint_path": str(self.checkpoint) if self.checkpoint else None,
                "capabilities": {
                    "global_map": _model.uses_global_map_observation(self.model)
                    if self.model else False,
                    "map_dag": _model.uses_map_dag(self.model) if self.model else False,
                    "card_language": _model.uses_card_language(self.model)
                    if self.model else False,
                    "combat_language": _model.uses_combat_language(self.model)
                    if self.model else False,
                },
                "embedding": self.embedding,
                "token_embeddings": self.token_embeddings,
                "card_catalog": {
                    card_id: {"name_zh": info.name_zh, "kind": info.kind}
                    for card_id, info in IRONCLAD_CARDS.items()
                } | {
                    card_id: {"name_zh": info["name_zh"], "kind": "status"}
                    for card_id, info in STATUS_CARDS.items()
                },
                "watch": self.watch.snapshot() if self.watch else None,
                "evaluation": self.eval_status,
                "error": self.error,
            }

    def load_checkpoint(self, path: Path, episodes: int):
        model = MaskablePPO.load(path, device="auto")
        model.set_random_seed(0)
        embedding = self._embedding(model)
        token_embeddings = self._token_embeddings(model)
        watch = WatchSession(model, self.model_lock, seed=0)
        with self.lock:
            old_watch = self.watch
            self.checkpoint = path
            self.model = model
            self.embedding = embedding
            self.token_embeddings = token_embeddings
            self.watch = watch
            self.error = None
        if old_watch:
            old_watch.close()
        self.start_evaluation(episodes=episodes, seed=0)

    def _embedding(self, model: MaskablePPO):
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
        indices = np.asarray([CARD_INDEX[name] for name in COMMON_CARDS])
        vectors = np.concatenate([base[indices], upgraded[indices]], axis=0)
        coords, ratio, norms = pca_project(vectors, dim=2, normalize=False)
        points = []
        for variant in (0, 1):
            for i, card_id in enumerate(COMMON_CARDS):
                row = variant * len(COMMON_CARDS) + i
                cost, damage, block = card_numbers(card_id, variant)
                points.append({
                    "card_id": card_id,
                    "name": COMMON_ZH[card_id] + ("+" if variant else ""),
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

    def _token_embeddings(self, model: MaskablePPO):
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
                "death_seeds": {}, "current_episode": 1,
                "current_steps": 0, "current_floor": 0,
                "current_phase": None, "current_action": None,
            }
        threading.Thread(
            target=self._evaluate,
            args=(token, model, episodes, int(seed)),
            daemon=True,
            name="dashboard-eval",
        ).start()

    def _evaluate(self, token: str, model: MaskablePPO, episodes: int, seed: int):
        env = SlayTheSpireEnv(
            global_map_view=_model.uses_global_map_observation(model),
            card_language=_model.uses_card_language(model),
            combat_language=_model.uses_combat_language(model),
        )
        death_floors: Counter[int] = Counter()
        death_seeds: dict[int, list[int]] = {}
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
                    result = APP.watch.reset(int(body.get("seed", 0)))
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
            elif path == "/api/eval/start":
                body = self._read_json()
                APP.start_evaluation(body.get("episodes", 50), body.get("seed", 0))
                self._send_json(APP.public_state()["evaluation"])
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
        if not filename.lower().endswith(".zip"):
            raise ValueError("only .zip checkpoints are accepted")
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
            if signature not in (b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08"):
                raise ValueError(
                    "uploaded file does not have a ZIP signature; re-download "
                    "the checkpoint and drag the .zip file itself"
                )
            os.replace(partial, target)
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
