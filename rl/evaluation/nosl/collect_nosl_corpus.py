"""Collect a public combat corpus and run a first NOSL Flat-MC comparison.

The collector follows a deterministic PPO trajectory, but every counterfactual
combat branch is re-determinized from an opaque NOSL nonce.  The live branch's
RNG state and hidden draw-pile order are never used as a future world-line.

Example (CPU smoke run)::

    python -m rl.evaluation.nosl.collect_nosl_corpus checkpoint.zip \
        --card-vector-checkpoint .tmp-vector-rule-v7-contract-v10-step10000/last.pt \
        --episodes 2 --max-states 64 --mc-samples 8 \
        --output runs/nosl/corpus.jsonl --summary runs/nosl/summary.json

The JSONL stores model observations, legal masks, seeds, and action prefixes.
It intentionally stores no raw BattleContext and no ordered hidden draw pile;
the prefix is the replay contract for a later independent replayer.
"""

from __future__ import annotations

from rl.policies.checkpoint import custom_objects as _load_custom_objects

import argparse
import copy
import hashlib
import json
import math
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks

from env.core.run_env import SlayTheSpireEnv
from env.core.combat_value import combat_end_hp
from rl.policies import model as model_helpers
from env.search.nosl_sampler import NOSLSampler

# Register custom policies before SB3 unpickles the checkpoint.
from rl.policies import pointer_policy as _pointer_policy  # noqa: F401
from rl.policies import split_policy as _split_policy  # noqa: F401


DEFAULT_CARD_VECTOR = Path(".tmp-vector-rule-v7-contract-v10-step10000/last.pt")
DEFAULT_MASTER_NONCE = 0x4E4F534C_7631
DEFAULT_MAX_EPISODE_STEPS = 6000


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _checkpoint_meta(path: Path) -> dict[str, Any]:
    match = re.search(r"(?:step|steps)(\d+)|(?:\d+)_steps", str(path))
    if match and match.group(1) is None:
        step_match = re.search(r"(\d+)_steps", str(path))
        global_steps = int(step_match.group(1)) if step_match else None
    else:
        global_steps = int(match.group(1)) if match else None
    return {
        "path": str(path.resolve()),
        "name": path.name,
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
        "global_steps": global_steps,
    }




def _public_canonical(env: SlayTheSpireEnv) -> dict[str, Any]:
    """Return a public debug view with every pile order removed.

    The current checkpoint has ``card_language=false`` and therefore does not
    expose ordered draw tokens.  Sorting the debug piles makes that invariant
    explicit in the artifact and protects future readers from using a hidden
    source-world order by accident.
    """
    canonical = copy.deepcopy(env.thick.obs_canonical())
    for pile in ("draw", "discard", "exhaust"):
        cards = canonical.get(pile)
        if isinstance(cards, list):
            canonical[pile] = sorted(
                cards,
                key=lambda card: (
                    str(card.get("id", "")) if isinstance(card, dict) else str(card),
                    int(card.get("up", 0)) if isinstance(card, dict) else 0,
                ),
            )
    return _jsonable(canonical)


def _clone_redeterminized(env, sampler: NOSLSampler, state_nonce: int, index: int):
    """Fork a thin combat env and replace its BattleContext with a NOSL clone."""
    branch = env.fork_combat()
    redetermined = sampler.clone_and_redeterminize(
        branch.thick.bc,
        state_nonce=state_nonce,
        simulation_index=index,
    )
    branch.thick.bc = redetermined
    branch._invalidate_backend_cache()
    return branch


def _terminal_hp(branch) -> float:
    return combat_end_hp(branch.thick.bc)


@dataclass(frozen=True)
class FlatMCResult:
    action: int
    means: dict[int, float]
    completed: dict[int, int]
    timeouts: dict[int, int]
    samples: int


def flat_mc_action(
    env: SlayTheSpireEnv,
    legal_mask: np.ndarray,
    *,
    sampler: NOSLSampler,
    state_nonce: int,
    samples: int,
    max_steps: int,
) -> FlatMCResult:
    """Choose the root action with the largest terminal post-combat HP.

    Rollout actions after the root are uniform over the current legal thin
    actions.  This is deliberately model-free: only simulator transitions,
    legal masks, and ``projected_exit_hp`` are used.
    """
    if samples < 1 or max_steps < 1:
        raise ValueError("samples and max_steps must be positive")
    legal = [int(action) for action in np.flatnonzero(legal_mask)]
    if not legal:
        raise ValueError("combat state has no legal action")

    means: dict[int, float] = {}
    completed: dict[int, int] = {}
    timeouts: dict[int, int] = {}
    for action in legal:
        values: list[float] = []
        timed_out = 0
        for sample in range(samples):
            # Include the root id in the state nonce so each candidate receives
            # an independent hidden future, even when sample numbers coincide.
            branch_nonce = sampler.nonce(
                state_nonce=(int(state_nonce) ^ (action * 0x9E3779B1)),
                simulation_index=sample,
            )
            branch = _clone_redeterminized(
                env, sampler, branch_nonce, sample
            )
            rng = np.random.default_rng(
                sampler.nonce(
                    state_nonce=(int(state_nonce) ^ (action * 0xD1B54A32)),
                    simulation_index=sample,
                )
            )
            terminated, next_mask = branch.search_step(action, validate=False)
            truncated = False
            steps = 1
            while not (terminated or truncated) and steps < max_steps:
                mask = np.asarray(next_mask, dtype=bool)
                choices = np.flatnonzero(mask)
                if len(choices) == 0:
                    truncated = True
                    break
                next_action = int(rng.choice(choices))
                terminated, next_mask = branch.search_step(
                    next_action, validate=False
                )
                steps += 1
            if terminated and branch.thick.bc.is_battle_over:
                values.append(_terminal_hp(branch))
            else:
                timed_out += 1
        completed[action] = len(values)
        timeouts[action] = timed_out
        means[action] = float(np.mean(values)) if values else float("-inf")

    best = max(legal, key=lambda action: (means[action], -action))
    return FlatMCResult(best, means, completed, timeouts, samples)


def _result_name(env: SlayTheSpireEnv, terminated: bool, truncated: bool) -> str:
    stats = env.thick.run_stats()
    if stats["act"] > 3 or stats["outcome"] == "PLAYER_VICTORY":
        return "win"
    if truncated:
        return "truncated"
    if terminated:
        return "death"
    return "unfinished"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--card-vector-checkpoint", type=Path, default=None)
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--seed", type=int, default=910000)
    parser.add_argument("--ascension", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-states", type=int, default=64)
    parser.add_argument("--state-stride", type=int, default=1)
    parser.add_argument(
        "--one-per-combat",
        action="store_true",
        help="Capture only the first decision state of each combat.",
    )
    parser.add_argument("--mc-samples", type=int, default=8)
    parser.add_argument("--mc-max-steps", type=int, default=96)
    parser.add_argument("--master-nonce", type=lambda value: int(value, 0), default=DEFAULT_MASTER_NONCE)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--progress-every", type=int, default=1)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.episodes < 1 or args.max_states < 1:
        raise ValueError("episodes and max-states must be positive")
    if args.state_stride < 1 or args.mc_samples < 1 or args.mc_max_steps < 1:
        raise ValueError("state-stride, mc-samples and mc-max-steps must be positive")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    card_vector = args.card_vector_checkpoint
    if card_vector is None and DEFAULT_CARD_VECTOR.is_file():
        card_vector = DEFAULT_CARD_VECTOR
    custom_objects = _load_custom_objects(args.checkpoint, card_vector)
    model = MaskablePPO.load(
        args.checkpoint, device=args.device, custom_objects=custom_objects
    )
    if model_helpers.uses_card_language(model):
        raise ValueError(
            "NOSL corpus requires card_language=false so hidden draw-pile "
            "order is not part of the public model observation"
        )
    env = SlayTheSpireEnv(
        ascension=args.ascension,
        global_map_view=model_helpers.uses_global_map_observation(model),
        card_language=False,
        combat_language=model_helpers.uses_combat_language(model),
        deck_size_observation=model_helpers.uses_deck_size_observation(model),
        neow_observation=model_helpers.uses_neow_observation(model),
        event_option_observation=model_helpers.uses_event_option_observation(model),
    )
    sampler = NOSLSampler(args.master_nonce)
    checkpoint_meta = _checkpoint_meta(args.checkpoint)
    card_meta = _checkpoint_meta(card_vector) if card_vector else None
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    episodes: list[dict[str, Any]] = []
    started = time.perf_counter()
    state_count = 0
    trajectory_combat_states = 0
    combat_index = 0
    active_combat = False

    try:
        for episode_index in range(args.episodes):
            episode_seed = args.seed + episode_index
            obs, info = env.reset(seed=episode_seed)
            prefix: list[int] = []
            active_combat = False
            terminated = truncated = False
            steps = 0
            while not (terminated or truncated) and steps < DEFAULT_MAX_EPISODE_STEPS:
                phase = info.get("phase")
                masks = get_action_masks(env)
                if phase == "combat":
                    trajectory_combat_states += 1
                    new_combat = not active_combat
                    if new_combat:
                        combat_index += 1
                    if (
                        state_count < args.max_states
                        and (
                            new_combat
                            if args.one_per_combat
                            else trajectory_combat_states % args.state_stride == 0
                        )
                    ):
                        action, _ = model.predict(
                            obs, action_masks=masks, deterministic=True
                        )
                        ppo_action = int(np.asarray(action).item())
                        state_nonce = sampler.nonce(
                            state_nonce=episode_seed,
                            simulation_index=steps,
                        )
                        mc = flat_mc_action(
                            env,
                            masks,
                            sampler=sampler,
                            state_nonce=state_nonce,
                            samples=args.mc_samples,
                            max_steps=args.mc_max_steps,
                        )
                        record = {
                            "schema_version": 2,
                            "record_id": state_count,
                            "episode": episode_index + 1,
                            "episode_seed": episode_seed,
                            "thin_step": steps,
                            "combat_index": combat_index,
                            "phase": phase,
                            "action_prefix": list(prefix),
                            "observation": _jsonable(obs),
                            "action_mask": _jsonable(np.asarray(masks, dtype=bool)),
                            "ppo_action": ppo_action,
                            "public_canonical": _public_canonical(env),
                            "run_stats": _jsonable(env.thick.run_stats()),
                            "act": int(env.thick.run_stats()["act"]),
                            "floor": int(env.thick.run_stats()["floor"]),
                            "encounter": str(
                                getattr(env.thick, "_encounter_name", lambda: "UNKNOWN")()
                            ),
                            "is_boss": int(env.thick.run_stats()["floor"]) in (16, 33, 50),
                            "capture_reason": (
                                "combat_start" if args.one_per_combat else "state_stride"
                            ),
                            "hidden_draw_order_omitted": True,
                            "nosl": {
                                "master_nonce": int(args.master_nonce),
                                "state_nonce": int(state_nonce),
                                "mc_samples": args.mc_samples,
                                "mc_max_steps": args.mc_max_steps,
                                "action": mc.action,
                                "means_terminal_hp": {
                                    str(action_id): (
                                        value if math.isfinite(value) else None
                                    )
                                    for action_id, value in mc.means.items()
                                },
                                "completed": {
                                    str(action_id): count
                                    for action_id, count in mc.completed.items()
                                },
                                "timeouts": {
                                    str(action_id): count
                                    for action_id, count in mc.timeouts.items()
                                },
                            },
                        }
                        records.append(record)
                        state_count += 1
                    action, _ = model.predict(
                        obs, action_masks=masks, deterministic=True
                    )
                else:
                    action, _ = model.predict(
                        obs, action_masks=masks, deterministic=True
                    )
                action_id = int(np.asarray(action).item())
                obs, _reward, terminated, truncated, info = env.step(action_id)
                prefix.append(action_id)
                steps += 1
                active_combat = phase == "combat"
            if steps >= DEFAULT_MAX_EPISODE_STEPS and not (terminated or truncated):
                truncated = True
            episodes.append(
                {
                    "episode": episode_index + 1,
                    "seed": episode_seed,
                    "steps": steps,
                    "result": _result_name(env, terminated, truncated),
                    "run_stats": _jsonable(env.thick.run_stats()),
                }
            )
            if args.progress_every and (episode_index + 1) % args.progress_every == 0:
                print(
                    f"[nosl] episode {episode_index + 1}/{args.episodes}: "
                    f"steps={steps} combat_states={trajectory_combat_states} "
                    f"records={state_count}",
                    flush=True,
                )
    finally:
        env.close()

    with args.output.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=True, allow_nan=False) + "\n")
    elapsed = time.perf_counter() - started
    summary = {
        "schema_version": 2,
        "created_at_unix": time.time(),
        "elapsed_seconds": elapsed,
        "checkpoint": checkpoint_meta,
        "card_vector_checkpoint": card_meta,
        "environment": {
            "ascension": args.ascension,
            "observation_card_language": False,
            "combat_language": model_helpers.uses_combat_language(model),
            "max_episode_steps": DEFAULT_MAX_EPISODE_STEPS,
            "sim_build": __import__("slaythespire").__file__,
        },
        "sampler": {
            "kind": "blake2b-nosl-v1",
            "master_nonce": int(args.master_nonce),
            "future_rng_source": "fresh RNG streams + canonicalized draw multiset",
        },
        "collection": {
            "episodes_requested": args.episodes,
            "episodes": episodes,
            "combat_states_seen": trajectory_combat_states,
            "records": len(records),
            "state_stride": args.state_stride,
            "one_per_combat": bool(args.one_per_combat),
            "combats_seen": combat_index,
            "mc_samples": args.mc_samples,
            "mc_max_steps": args.mc_max_steps,
        },
        "output": str(args.output.resolve()),
    }
    args.summary.write_text(
        json.dumps(summary, ensure_ascii=True, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(f"[nosl] wrote {len(records)} records to {args.output.resolve()}")
    print(f"[nosl] wrote summary to {args.summary.resolve()}")


if __name__ == "__main__":
    main()
