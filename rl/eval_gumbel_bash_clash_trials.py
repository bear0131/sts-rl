"""Repeated Bash/Clash trials across public states and NOSL worlds.

Each accepted state has both Bash and Clash in the opening hand.  PUCT and
Gumbel searches use the same root nonce for a paired comparison; changing the
nonce changes the hidden future RNG without changing the public fixture.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean, stdev

from sb3_contrib import MaskablePPO

from env.core import _BUILD  # noqa: F401
from env.core import action_space as A
from rl.collect_nosl_corpus import _load_custom_objects
from rl.eval_nosl_order_sensitivity import (
    SCENARIOS,
    _card_dict,
    _jsonable,
    _make_env,
    _root_policy,
    _run_search,
)
from rl.nosl_sampler import NOSLSampler


FILLER_PROFILES = {
    "two_cards": (),
    # Keep the future draw pile non-empty while making the opening hand a
    # strict two-choice root.  Bludgeon is an Attack (so Clash remains legal)
    # but costs 3, hence it is masked at the controlled energy=2 root.  The
    # remaining ordinary cards stay in the draw pile and preserve NOSL future
    # variation after the first card.
    "strict_root_normal_draw": (
        "BLUDGEON", "BLUDGEON", "BLUDGEON", "BLUDGEON", "BLUDGEON",
        "BLOOD_FOR_BLOOD", "BLOOD_FOR_BLOOD", "BLOOD_FOR_BLOOD",
        "BLOOD_FOR_BLOOD", "BLOOD_FOR_BLOOD",
        "STRIKE_RED", "DEFEND_RED", "IRON_WAVE", "POMMEL_STRIKE",
        "SHRUG_IT_OFF", "ANGER", "CLEAVE", "TRUE_GRIT", "FLEX",
        "THUNDERCLAP",
    ),
    "strikes": ("STRIKE_RED",) * 8,
    "mixed_attacks": (
        "STRIKE_RED", "CLEAVE", "IRON_WAVE", "BODY_SLAM",
        "HEADBUTT", "POMMEL_STRIKE", "TWIN_STRIKE", "THUNDERCLAP",
    ),
}


def _hand(env) -> list[dict[str, object]]:
    return [_card_dict(card) for card in env.thick.bc.to_dict()["hand"]]


def _has_pair(env, hand: list[dict[str, object]]) -> bool:
    ids = {str(card["id"]).upper() for card in hand}
    if "BASH" not in ids or "CLASH" not in ids:
        return False
    # Clash is only a meaningful candidate when the simulator considers it
    # legal; this filters mixed decks whose opening hand also contains a Skill.
    clash_slot = next(
        index for index, card in enumerate(hand)
        if str(card["id"]).upper() == "CLASH"
    )
    return bool(env.action_masks()[A.seg_base("PLAY_CARD") + clash_slot])


def _legal_cards(env, hand: list[dict[str, object]]) -> list[str]:
    mask = env.action_masks()
    play_base = A.seg_base("PLAY_CARD")
    return [
        str(card["id"]).upper()
        for slot, card in enumerate(hand)
        if bool(mask[play_base + slot])
    ]


def _make_trial_env(model, *, seed: int, swapped: bool, fillers):
    env = _make_env(
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


def _classify(action: str) -> str:
    upper = str(action).upper()
    if "BASH" in upper:
        return "bash"
    if "CLASH" in upper:
        return "clash"
    return "other"


def _root_qs(search: dict[str, object]) -> tuple[float | None, float | None]:
    root = search.get("tree_root") or {}
    bash = None
    clash = None
    for child in root.get("children", ()):
        label = str(child.get("label", "")).upper()
        if "BASH" in label:
            bash = float(child.get("value", 0.0))
        elif "CLASH" in label:
            clash = float(child.get("value", 0.0))
    return bash, clash


def _direct_stats(model, env, hand):
    _, probs, value, _ = _root_policy(model, env)
    by_card = {str(card["id"]).upper(): [] for card in hand}
    for slot, card in enumerate(hand):
        by_card.setdefault(str(card["id"]).upper(), []).append(slot)
    # The fixture has one Bash and one Clash.  Keep the lookup slot-aware so
    # repeated copies of filler cards do not affect the reported probabilities.
    play_base = 1
    bash_slot = by_card["BASH"][0]
    clash_slot = by_card["CLASH"][0]
    legal_play = [1 + slot for slot in range(len(hand))]
    direct_slot = max(legal_play, key=lambda action: float(probs[action]))
    return {
        "value": float(value),
        "bash_prob": float(probs[play_base + bash_slot]),
        "clash_prob": float(probs[play_base + clash_slot]),
        "argmax": _classify(f"PLAY:{hand[direct_slot - play_base]['id']}"),
    }


def _summary(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, int], list[dict[str, object]]] = {}
    for row in rows:
        key = (str(row["method"]), int(row["simulations"]))
        groups.setdefault(key, []).append(row)
    result = []
    for (method, simulations), items in sorted(groups.items()):
        n = len(items)
        counts = {
            name: sum(1 for item in items if item["selected"] == name)
            for name in ("bash", "clash", "other")
        }
        deltas = [
            float(item["bash_q"] - item["clash_q"])
            for item in items
            if item.get("bash_q") is not None and item.get("clash_q") is not None
        ]
        bash_visits = [float(item["bash_visits"]) for item in items]
        clash_visits = [float(item["clash_visits"]) for item in items]
        delta_mean = mean(deltas) if deltas else None
        delta_ci = None
        if len(deltas) > 1:
            delta_ci = 1.96 * stdev(deltas) / math.sqrt(len(deltas))
        result.append({
            "method": method,
            "simulations": simulations,
            "n": n,
            "bash_selected": counts["bash"],
            "clash_selected": counts["clash"],
            "other_selected": counts["other"],
            "bash_selection_rate": counts["bash"] / n if n else 0.0,
            "mean_bash_visits": mean(bash_visits) if bash_visits else None,
            "mean_clash_visits": mean(clash_visits) if clash_visits else None,
            "mean_q_delta_bash_minus_clash": delta_mean,
            "q_delta_95ci_half_width": delta_ci,
        })
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--seed-stop", type=int, default=500)
    parser.add_argument("--max-states", type=int, default=30)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--profile", choices=tuple(FILLER_PROFILES),
                        default="two_cards")
    parser.add_argument("--simulations", nargs="+", type=int,
                        default=[128, 256, 512])
    parser.add_argument("--depth", type=int, default=0)
    parser.add_argument("--leaf-batch", type=int, default=64)
    parser.add_argument("--gumbel-candidates", type=int, default=0)
    parser.add_argument("--include-swapped", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.seed_stop <= args.seed_start:
        raise ValueError("--seed-stop must be greater than --seed-start")

    model = MaskablePPO.load(
        args.checkpoint,
        device="cpu",
        custom_objects=_load_custom_objects(args.checkpoint, None),
    )
    fillers = FILLER_PROFILES[args.profile]
    states = []
    layouts = (False, True) if args.include_swapped else (False,)
    for seed in range(args.seed_start, args.seed_stop):
        for swapped in layouts:
            env = _make_trial_env(
                model,
                swapped=swapped,
                seed=seed,
                fillers=fillers,
            )
            hand = _hand(env)
            if not _has_pair(env, hand):
                del env
                continue
            legal_cards = _legal_cards(env, hand)
            if args.profile in {"two_cards", "strict_root_normal_draw"} and sorted(legal_cards) != [
                "BASH", "CLASH"
            ]:
                raise RuntimeError(
                    f"strict root has unexpected legal cards: {legal_cards}"
                )
            raw = env.thick.bc.to_dict()
            states.append({
                "seed": int(seed),
                "swapped": bool(swapped),
                "energy": int(raw["player"]["energy"]),
                "legal_cards": legal_cards,
                "hand": hand,
                "draw": [_card_dict(card) for card in raw["draw"]],
            })
            del env
            if len(states) >= int(args.max_states):
                break
        if len(states) >= int(args.max_states):
            break

    if not states:
        raise RuntimeError("no seed produced an opening Bash+Clash hand")

    sampler = NOSLSampler(0x4E4F534C_7631)
    rows = []
    for state_index, state_meta in enumerate(states):
        env = _make_trial_env(
            model,
            swapped=bool(state_meta["swapped"]),
            seed=int(state_meta["seed"]),
            fillers=fillers,
        )
        hand = _hand(env)
        direct = _direct_stats(model, env, hand)
        for repetition in range(max(1, int(args.repeats))):
            nonce = sampler.nonce(
                state_nonce=(int(state_meta["seed"]) << 1) |
                int(bool(state_meta["swapped"])),
                simulation_index=200_000 + state_index * 1000 + repetition,
            )
            for simulations in args.simulations:
                for method, root_mode in (("puct", 0), ("gumbel", 1)):
                    result = _run_search(
                        model,
                        env,
                        simulations=int(simulations),
                        depth=int(args.depth),
                        nonce=int(nonce),
                        leaf_batch=int(args.leaf_batch),
                        prior_mode="model",
                        root_search_mode=root_mode,
                        gumbel_candidates=int(args.gumbel_candidates),
                    )
                    search = result["search"]
                    bash_q, clash_q = _root_qs(search)
                    visits = search["root_visits"]
                    rows.append({
                        **state_meta,
                        "state_index": state_index,
                        "repetition": repetition,
                        "nonce": int(nonce),
                        "method": method,
                        "simulations": int(simulations),
                        "depth": int(args.depth),
                        "selected_action": search["action"],
                        "selected": _classify(search["action"]),
                        "bash_visits": int(sum(
                            int(v) for k, v in visits.items()
                            if "BASH" in str(k).upper()
                        )),
                        "clash_visits": int(sum(
                            int(v) for k, v in visits.items()
                            if "CLASH" in str(k).upper()
                        )),
                        "bash_q": bash_q,
                        "clash_q": clash_q,
                        "root_value": float(search["root_value"]),
                        "direct_bash_prob": direct["bash_prob"],
                        "direct_clash_prob": direct["clash_prob"],
                        "direct_value": direct["value"],
                    })

    payload = {
        "schema_version": 1,
        "checkpoint": str(args.checkpoint.resolve()),
        "scenario": "bash_plus_clash",
        "profile": args.profile,
        "seed_range": [int(args.seed_start), int(args.seed_stop)],
        "states": states,
        "repeats": int(args.repeats),
        "simulations": [int(x) for x in args.simulations],
        "depth": int(args.depth),
        "leaf_batch": int(args.leaf_batch),
        "gumbel_candidates": int(args.gumbel_candidates),
        "rows": rows,
        "summary": _summary(rows),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(_jsonable(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        f"states={len(states)} rows={len(rows)} output={args.output.resolve()}"
    )
    for item in payload["summary"]:
        print(json.dumps(item, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
