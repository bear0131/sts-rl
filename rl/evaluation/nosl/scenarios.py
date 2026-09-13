"""Reusable fixed-combat fixtures and search diagnostics; no command-line entry."""
from __future__ import annotations
from typing import Any
import numpy as np
from env.core import _BUILD  # noqa: F401
from env.core import action_space as A
from env.core.game_env import SimGameEnv
from env.core.run_env import SlayTheSpireEnv
from rl.policies import model as model_helpers
from rl.search.native import NativeSearchConfig, search as native_search, forward_root
from configs.eval.nosl_scenarios import FILLER_CARDS, SCENARIOS
import slaythespire as sts


def jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [jsonable(v) for v in value]
    return value


def card_dict(card) -> dict[str, Any]:
    if isinstance(card, dict):
        return dict(card)
    values = list(card)
    result = {"id": str(values[0]), "up": int(values[1])}
    if len(values) > 2:
        result["cost"] = int(values[2])
    return result


def env_for_model(model) -> SlayTheSpireEnv:
    return SlayTheSpireEnv(
        ascension=0,
        global_map_view=model_helpers.uses_global_map_observation(model),
        card_language=model_helpers.uses_card_language(model),
        combat_language=model_helpers.uses_combat_language(model),
        deck_size_observation=model_helpers.uses_deck_size_observation(model),
        neow_observation=model_helpers.uses_neow_observation(model),
        event_option_observation=model_helpers.uses_event_option_observation(model),
    )


def make_env(
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
    env = env_for_model(model)
    env.thick = thick
    env._invalidate_backend_cache()
    return env


def forward(model, observations):
    return forward_root(model, observations)


def root_policy(model, env):
    obs = env._obs()
    logits, values = forward(model, obs)
    raw_logits = np.asarray(logits[0], dtype=np.float64)
    mask = np.asarray(env.action_masks(), dtype=bool)
    probs = np.zeros(raw_logits.size, dtype=np.float64)
    legal = np.flatnonzero(mask)
    selected = raw_logits[legal] - np.max(raw_logits[legal])
    weights = np.exp(np.clip(selected, -80.0, 80.0))
    probs[legal] = weights / float(weights.sum())
    return raw_logits, probs, float(values[0]), obs


def action_label(action: int, hand: list[dict[str, Any]]) -> str:
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


def root_table(values, hand):
    return {
        action_label(index, hand): int(values[index])
        for index in np.flatnonzero(np.asarray(values) > 0)
    }


def prob_table(values, hand):
    return {
        action_label(index, hand): float(values[index])
        for index in np.flatnonzero(np.asarray(values) > 0)
    }


def run_search(
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
    root_logits, direct_probs, direct_value, obs = root_policy(model, env)
    if prior_mode not in {"model", "uniform", "none"}:
        raise ValueError(f"unknown prior mode: {prior_mode}")
    config = NativeSearchConfig(
        simulations=simulations,
        depth=depth,
        c_puct=c_puct,
        prior_temperature=prior_temperature,
        root_search_mode=root_search_mode,
        gumbel_candidates=gumbel_candidates,
        gumbel_scale=gumbel_scale,
        gumbel_value_scale=gumbel_value_scale,
        gumbel_max_visit_init=gumbel_max_visit_init,
        gumbel_rescale_q=gumbel_rescale_q,
        gumbel_use_mixed_value=gumbel_use_mixed_value,
        gumbel_interior_uniformity=gumbel_interior_uniformity,
        gumbel_interior_min_visits=gumbel_interior_min_visits,
        gumbel_interior_coverage_fraction=gumbel_interior_coverage_fraction,
        paper_full_gumbel=paper_full_gumbel,
        leaf_batch_size=leaf_batch,
    )
    result = native_search(model, env, obs, config, nonce=nonce,
                           root_outputs=(root_logits, direct_value), prior_mode=prior_mode)
    leaf_count, batches = result["leaf_evaluations"], result["leaf_batches"]
    adapted = result
    hand = [card_dict(card) for card in env.thick.bc.to_dict()["hand"]]
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
            "action_probs": prob_table(direct_probs, hand),
            "argmax": action_label(int(np.argmax(direct_probs)), hand),
        },
        "search": {
            "action": action_label(int(result["action"]), hand),
            "action_id": int(result["action"]),
            "native_action_bits": int(result.get("native_action_bits", 0)),
            "root_value": float(result["root_value"]),
            "root_visits": root_table(visits, hand),
            "tree_root": tree,
            "tree_after_bash": tree_after_bash,
            # Keep both the raw visit distribution and Full-Gumbel's
            # completed-Q policy target.  The two are intentionally
            # different: visits are a budget-allocation diagnostic, while
            # `improved_policy` is the target used by the AZ learner.
            "root_visit_policy": prob_table(
                adapted.get("root_visit_policy", np.zeros(0)), hand
            ),
            "improved_policy": prob_table(
                adapted.get("improved_policy", np.zeros(0)), hand
            ),
            "used_completed_policy": bool(
                adapted.get("used_completed_policy", False)
            ),
        },
        "leaf_evaluations": int(leaf_count),
        "leaf_batches": int(batches),
    }


def combat_hand(env) -> list[dict[str, object]]:
    return [card_dict(card) for card in env.thick.bc.to_dict()["hand"]]


def legal_cards(env, hand: list[dict[str, object]]) -> list[str]:
    mask = env.action_masks()
    play_base = A.seg_base("PLAY_CARD")
    return [
        str(card["id"]).upper()
        for slot, card in enumerate(hand)
        if bool(mask[play_base + slot])
    ]


def make_trial_env(model, *, seed: int, swapped: bool, fillers):
    env = make_env(
        model,
        SCENARIOS["bash_plus_clash"],
        swapped=swapped,
        seed=seed,
        filler_cards=fillers,
    )
    # The controlled ordering test starts with exactly two energy. Bash spends
    # both points and Clash remains playable for zero; in the reverse order,
    # Bash is still playable after Clash.
    env.thick.bc.set_player_energy(2)
    env._invalidate_backend_cache()
    return env


def execute_selected(env, search: dict[str, Any]) -> None:
    """Apply the complete native card+target action returned by the search."""
    bits = int(search["search"]["native_action_bits"])
    env.thick.bc.execute(sts.Action(bits))
    env._invalidate_backend_cache()
