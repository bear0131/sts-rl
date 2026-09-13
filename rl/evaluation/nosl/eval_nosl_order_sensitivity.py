"""Measure C++ NOSL-IS-AZ sensitivity to hand-slot order and search depth.

The fixtures are real simulator combats.  The two relevant cards occupy the
same two deck positions and are swapped between layouts; all other cards,
seed, encounter, HP, and draw-pile composition stay fixed.  The remaining
deck is made of varied ordinary Ironclad cards so the next draw can change the
future damage/block value; the layouts differ only in the two target slots.

Example::

    python -m rl.evaluation.nosl.eval_nosl_order_sensitivity \
        artifacts/latest-nosl-az-s512-20260905/nosl_az_614400_steps.zip \
        --simulations 128 --depths 1 2 3 4 --depth0-simulations 512 \
        --output .tmp-nosl-order-sensitivity.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sb3_contrib import MaskablePPO

from env.core import _BUILD  # noqa: F401
from env.core import action_space as A
from env.core.game_env import SimGameEnv
from env.core.run_env import SlayTheSpireEnv, unpack_nosl_observation_buffer
from rl.policies import model as model_helpers
from rl.evaluation.nosl.collect_nosl_corpus import _load_custom_objects
from env.search.nosl_sampler import NOSLSampler

import slaythespire as sts

from rl.policies import pointer_policy as _pointer_policy  # noqa: F401
from rl.policies import split_policy as _split_policy  # noqa: F401


from configs.eval.nosl_scenarios import FILLER_CARDS, SCENARIOS


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(v) for v in value]
    return value


def _card_dict(card) -> dict[str, Any]:
    if isinstance(card, dict):
        return dict(card)
    values = list(card)
    result = {"id": str(values[0]), "up": int(values[1])}
    if len(values) > 2:
        result["cost"] = int(values[2])
    return result


def _env_for_model(model) -> SlayTheSpireEnv:
    return SlayTheSpireEnv(
        ascension=0,
        global_map_view=model_helpers.uses_global_map_observation(model),
        card_language=model_helpers.uses_card_language(model),
        combat_language=model_helpers.uses_combat_language(model),
        deck_size_observation=model_helpers.uses_deck_size_observation(model),
        neow_observation=model_helpers.uses_neow_observation(model),
        event_option_observation=model_helpers.uses_event_option_observation(model),
    )


def _make_env(
    model,
    pair,
    swapped: bool,
    seed: int,
    filler_cards=FILLER_CARDS,
) -> SlayTheSpireEnv:
    ordered_pair = tuple(pair[::-1] if swapped else pair)
    deck = list(ordered_pair)
    deck.extend((card, 0) for card in filler_cards)
    gc = sts.GameContext(sts.CharacterClass.IRONCLAD, int(seed), 0)
    gc.floor_num = 1
    gc.cur_hp = gc.max_hp
    gc.set_deck(deck)
    thick = SimGameEnv()
    thick.gc = gc
    thick.bc = sts.create_battle(gc, "Cultist")
    thick._prev_hp = gc.cur_hp
    env = _env_for_model(model)
    env.thick = thick
    env._invalidate_backend_cache()
    return env


def _forward(model, observations):
    policy = model.policy
    expected = getattr(getattr(model, "observation_space", None), "spaces", None)
    if expected:
        observations = {
            key: value for key, value in observations.items() if key in expected
        }
    tensor, _ = policy.obs_to_tensor(observations)
    core = getattr(policy, "core", None)
    if core is None or not hasattr(core, "actor_critic"):
        raise RuntimeError("checkpoint does not expose the AlphaZero actor_critic API")
    with torch.inference_mode():
        logits, values = core.actor_critic(tensor)
    return (
        logits.detach().float().cpu().numpy(),
        values.detach().float().reshape(-1).cpu().numpy(),
    )


def _root_policy(model, env):
    obs = env._obs()
    logits, values = _forward(model, obs)
    raw_logits = np.asarray(logits[0], dtype=np.float64)
    mask = np.asarray(env.action_masks(), dtype=bool)
    probs = np.zeros(raw_logits.size, dtype=np.float64)
    legal = np.flatnonzero(mask)
    selected = raw_logits[legal] - np.max(raw_logits[legal])
    weights = np.exp(np.clip(selected, -80.0, 80.0))
    probs[legal] = weights / float(weights.sum())
    return raw_logits, probs, float(values[0]), obs


def _action_label(action: int, hand: list[dict[str, Any]]) -> str:
    action = int(action)
    if action == A.seg_base("END_TURN"):
        return "END_TURN"
    play = A.seg_base("PLAY_CARD")
    if play <= action < play + A.HAND_SLOTS:
        slot = action - play
        card = hand[slot]["id"] if slot < len(hand) else "?"
        up = int(hand[slot].get("up", 0)) if slot < len(hand) else 0
        return f"PLAY:{card}{'+' if up else ''}@{slot}"
    return f"ACTION:{action}"


def _root_table(values, hand):
    return {
        _action_label(index, hand): int(values[index])
        for index in np.flatnonzero(np.asarray(values) > 0)
    }


def _prob_table(values, hand):
    return {
        _action_label(index, hand): float(values[index])
        for index in np.flatnonzero(np.asarray(values) > 0)
    }


def _run_search(
    model,
    env,
    *,
    simulations: int,
    depth: int,
    nonce: int,
    leaf_batch: int,
    prior_mode: str = "model",
    root_search_mode: int = 0,
    gumbel_candidates: int = 0,
    c_puct: float = 1.5,
    prior_temperature: float = 1.0,
    gumbel_scale: float = 1.0,
    gumbel_value_scale: float = 0.1,
    gumbel_max_visit_init: float = 50.0,
    gumbel_rescale_q: bool = True,
    gumbel_use_mixed_value: bool = True,
    gumbel_interior_uniformity: float = 0.0,
    gumbel_interior_min_visits: int = 1,
    gumbel_interior_coverage_fraction: float = 0.0,
    paper_full_gumbel: bool = False,
):
    root_logits, direct_probs, direct_value, obs = _root_policy(model, env)
    if prior_mode not in {"model", "uniform", "none"}:
        raise ValueError(f"unknown prior mode: {prior_mode}")
    search_logits = root_logits
    c_puct = float(c_puct)
    if prior_mode in {"uniform", "none"}:
        # Equal logits produce a uniform legal-action prior in the native
        # masked softmax.  ``none`` additionally removes the PUCT prior term,
        # which is a useful strict ablation but is not the usual AZ setup.
        search_logits = np.zeros_like(root_logits)
    if prior_mode == "none":
        c_puct = 0.0
    response = env.nosl_is_az_cpp_begin(
        search_logits,
        state_nonce=int(nonce),
        simulations=int(simulations),
        depth=int(depth),
        c_puct=c_puct,
        leaf_batch_size=int(leaf_batch),
        dirichlet_alpha=0.0,
        dirichlet_epsilon=0.0,
        root_value=float(direct_value),
        prior_temperature=float(prior_temperature),
        root_search_mode=int(root_search_mode),
        gumbel_candidates=int(gumbel_candidates),
        gumbel_scale=float(gumbel_scale),
        gumbel_value_scale=float(gumbel_value_scale),
        gumbel_max_visit_init=float(gumbel_max_visit_init),
        gumbel_rescale_q=bool(gumbel_rescale_q),
        gumbel_use_mixed_value=bool(gumbel_use_mixed_value),
        gumbel_interior_uniformity=float(gumbel_interior_uniformity),
        gumbel_interior_min_visits=int(gumbel_interior_min_visits),
        gumbel_interior_coverage_fraction=float(
            gumbel_interior_coverage_fraction
        ),
        paper_full_gumbel=bool(paper_full_gumbel),
        stochastic=False,
        decision_index=0,
    )
    leaf_count = 0
    batches = 0
    while not response.get("done", False):
        leaves = response.get("leaf_meta", response.get("leaves", ()))
        if not leaves:
            raise RuntimeError("native search returned no leaves before completion")
        batches += 1
        leaf_count += len(leaves)
        packed = unpack_nosl_observation_buffer(response)
        if packed is not None:
            observations = {
                key: np.ascontiguousarray(value) for key, value in packed.items()
            }
        else:
            observations = {
                key: np.stack([np.asarray(leaf["observation"][key]) for leaf in leaves])
                for key in leaves[0]["observation"]
            }
        leaf_logits, leaf_values = _forward(model, observations)
        updates = []
        for index, leaf in enumerate(leaves):
            if not leaf.get("needs_policy", True):
                policy = np.zeros(0)
            elif prior_mode == "model":
                policy = leaf_logits[index]
            else:
                # The ablation applies to every newly expanded node, not just
                # the root.  Values still come from the trained value head.
                policy = np.zeros_like(leaf_logits[index])
            updates.append((int(leaf["request_id"]), float(leaf_values[index]), policy))
        env.nosl_is_az_cpp_backprop_batch(updates)
        response = env.nosl_is_az_cpp_leaf_batch(int(leaf_batch))

    result = response["result"]
    adapted = result
    hand = [_card_dict(card) for card in env.thick.bc.to_dict()["hand"]]
    visits = np.asarray(result["root_visits"], dtype=np.int64)
    tree = env.nosl_is_az_cpp_tree_snapshot((), 64)
    bash_edge = next(
        (
            child for child in tree.get("children", ())
            if "Bash" in str(child.get("label", ""))
        ),
        None,
    )
    tree_after_bash = None
    if bash_edge is not None:
        tree_after_bash = env.nosl_is_az_cpp_tree_snapshot(
            (int(bash_edge["edge_index"]),), 64
        )
    return {
        "depth": int(depth),
        "simulations": int(simulations),
        "prior_mode": prior_mode,
        "root_search_mode": (
            "full_gumbel" if int(root_search_mode) == 2
            else "gumbel_sequential_halving" if int(root_search_mode) == 1
            else "puct"
        ),
        "gumbel_candidates": int(gumbel_candidates),
        "c_puct": float(c_puct),
        "prior_temperature": float(prior_temperature),
        "gumbel_scale": float(gumbel_scale),
        "gumbel_value_scale": float(gumbel_value_scale),
        "gumbel_max_visit_init": float(gumbel_max_visit_init),
        "gumbel_rescale_q": bool(gumbel_rescale_q),
        "gumbel_use_mixed_value": bool(gumbel_use_mixed_value),
        "direct_policy": {
            "value": direct_value,
            "action_probs": _prob_table(direct_probs, hand),
            "argmax": _action_label(int(np.argmax(direct_probs)), hand),
        },
        "search": {
            "action": _action_label(int(result["action"]), hand),
            "action_id": int(result["action"]),
            "native_action_bits": int(result.get("native_action_bits", 0)),
            "root_value": float(result["root_value"]),
            "root_visits": _root_table(visits, hand),
            "tree_root": tree,
            "tree_after_bash": tree_after_bash,
            # Keep both the raw visit distribution and Full-Gumbel's
            # completed-Q policy target.  The two are intentionally
            # different: visits are a budget-allocation diagnostic, while
            # `improved_policy` is the target used by the AZ learner.
            "root_visit_policy": _prob_table(
                adapted.get("root_visit_policy", np.zeros(0)), hand
            ),
            "improved_policy": _prob_table(
                adapted.get("improved_policy", np.zeros(0)), hand
            ),
            "used_completed_policy": bool(
                adapted.get("used_completed_policy", False)
            ),
        },
        "leaf_evaluations": int(leaf_count),
        "leaf_batches": int(batches),
    }


def _build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--seed", type=int, default=36)
    parser.add_argument("--simulations", type=int, default=128)
    parser.add_argument("--depths", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument("--depth0-simulations", type=int, default=512)
    parser.add_argument("--leaf-batch", type=int, default=64)
    parser.add_argument(
        "--prior-mode",
        choices=("model", "uniform", "none"),
        default="model",
        help=(
            "root prior ablation: model logits (default), uniform legal prior, "
            "or none (uniform plus c_puct=0)"
        ),
    )
    parser.add_argument(
        "--root-search-mode",
        choices=("puct", "gumbel"),
        default="puct",
        help="root budget allocator; descendants remain PUCT",
    )
    parser.add_argument(
        "--gumbel-candidates",
        type=int,
        default=0,
        help="number of Gumbel-ranked root candidates; 0 means all legal roots",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main():
    args = _build_parser().parse_args()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    model = MaskablePPO.load(
        args.checkpoint,
        device=args.device,
        custom_objects=_load_custom_objects(args.checkpoint, None),
    )
    output = {
        "schema_version": 1,
        "checkpoint": str(args.checkpoint.resolve()),
        "seed": int(args.seed),
        "encounter": "Cultist",
        "draw_size": 15,
        "draw_is_non_unique": True,
        "scenarios": {},
    }
    started = time.perf_counter()
    for scenario, pair in SCENARIOS.items():
        rows = []
        fillers = (
            ("STRIKE_RED",) * 8
            if scenario == "bash_plus_clash" else FILLER_CARDS
        )
        for swapped in (False, True):
            env = _make_env(model, pair, swapped, args.seed, fillers)
            raw = env.thick.bc.to_dict()
            hand = [_card_dict(card) for card in raw["hand"]]
            settings = [(int(args.simulations), depth) for depth in args.depths]
            settings.append((int(args.depth0_simulations), 0))
            for simulations, depth in settings:
                print(
                    f"[order] {scenario} layout={'swapped' if swapped else 'base'} "
                    f"depth={depth} simulations={simulations} "
                    f"prior={args.prior_mode}", flush=True
                )
                row = {
                    "scenario": scenario,
                    "layout": "swapped" if swapped else "base",
                    "seed": int(args.seed),
                    "hand": hand,
                    "draw": [_card_dict(card) for card in raw["draw"]],
                    "draw_size": len(raw["draw"]),
                }
                search = _run_search(
                    model,
                    env,
                    simulations=simulations,
                    depth=depth,
                    prior_mode=args.prior_mode,
                    nonce=NOSLSampler(0x4E4F534C_7631).nonce(
                        state_nonce=args.seed,
                        simulation_index=(len(rows) + (1 if swapped else 0) * 10000),
                    ),
                    leaf_batch=args.leaf_batch,
                    root_search_mode=(1 if args.root_search_mode == "gumbel" else 0),
                    gumbel_candidates=args.gumbel_candidates,
                )
                row.update(search)
                rows.append(row)
            del env
        output["scenarios"][scenario] = rows
    output["elapsed_seconds"] = time.perf_counter() - started
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(_jsonable(output), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[order] wrote {args.output.resolve()} elapsed={output['elapsed_seconds']:.1f}s")


if __name__ == "__main__":
    main()
