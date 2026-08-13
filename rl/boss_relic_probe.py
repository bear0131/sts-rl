from __future__ import annotations

import argparse
from collections import defaultdict
from itertools import combinations, permutations
from pathlib import Path
import math
import time

import numpy as np
import torch
from sb3_contrib import MaskablePPO
from stable_baselines3.common.utils import obs_as_tensor

from bridge.schemas.combat_state import _sim_card_id
from env.core import action_space as A
from env.core import encode
from env.core import vocab as V
from env.core.factories import _build_gauntlet_gc

# Register custom checkpoint classes before SB3 restores the policy.
from rl import model as _model  # noqa: F401
from rl import pointer_policy as _pointer_policy  # noqa: F401
from rl import split_policy as _split_policy  # noqa: F401


ACT1_BOSSES = ["SLIME_BOSS", "THE_GUARDIAN", "HEXAGHOST"]
BOSS_POOL = [
    "Fusion Hammer",
    "Velvet Choker",
    "Runic Dome",
    "SlaversCollar",
    "Snecko Eye",
    "Pandora's Box",
    "Cursed Key",
    "Busted Crown",
    "Ectoplasm",
    "Tiny House",
    "Sozu",
    "Philosopher's Stone",
    "Astrolabe",
    "Black Star",
    "SacredBark",
    "Empty Cage",
    "Runic Pyramid",
    "Calling Bell",
    "Coffee Dripper",
    "Black Blood",
    "Mark of Pain",
    "Runic Cube",
]


def obs_flags(space):
    keys = set(getattr(space, "spaces", {}))
    return dict(
        global_map_view="map_full_rooms" in keys,
        card_language="draw_card_ids" in keys,
        combat_language="player_status_language" in keys,
        deck_size_observation="deck_size" in keys,
        neow_observation="neow_bonus_ids" in keys,
        event_option_observation="event_option_ids" in keys,
    )


def make_background(rng: np.random.Generator) -> dict:
    gc = _build_gauntlet_gc(rng, 16, 0, (0.25, 1.0))
    boss_pool = set(BOSS_POOL)
    held = [r for r in gc.relic_ids() if r not in boss_pool]
    if "Burning Blood" not in held:
        held.insert(0, "Burning Blood")
    return {
        "character": "IRONCLAD",
        "ascension": 0,
        "act": 1,
        "floor": 17,
        "boss": str(rng.choice(ACT1_BOSSES)),
        "relics": held,
        "potions": list(gc.overworld_potions),
        "gold": int(rng.integers(0, 301)),
        "deck": [(_sim_card_id(name), up) for name, up in gc.deck_ids()],
    }


def make_base_obs(global_block: dict, flags: dict) -> dict:
    canon = {
        "phase": "boss_relic",
        "arg_step": "root",
        "options": {},
        "global": {**global_block, "boss_relics": []},
    }
    return encode.encode(canon, **flags)


def add_boss_relics(base_obs: dict, offered: tuple[str, str, str]) -> dict:
    obs = dict(base_obs)
    boss = base_obs["boss_relic_ids"].copy()
    for i, relic in enumerate(offered):
        boss[i] = V.RELIC_VOCAB_INDEX[relic]
    obs["boss_relic_ids"] = boss
    return obs


def batch_encode(model, obs_list, offered_list):
    batch = {k: np.stack([obs[k] for obs in obs_list], axis=0) for k in obs_list[0]}
    obs_t = obs_as_tensor(batch, model.device)
    base = A.seg_base("BOSS_RELIC")
    with torch.no_grad():
        logits = model.policy.core.action_logits(obs_t)
        probs = torch.softmax(logits[:, base : base + 3], dim=-1).detach().cpu().numpy()
    return probs, offered_list


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--checkpoint",
        default=r"C:\Users\Tony2\Downloads\maskable_ppo_6400000_steps.zip",
    )
    ap.add_argument("--backgrounds", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=20260810)
    args = ap.parse_args()

    ckpt = Path(args.checkpoint)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MaskablePPO.load(str(ckpt), device=device)
    model.policy.eval()
    flags = obs_flags(model.observation_space)

    combos = list(combinations(BOSS_POOL, 3))
    total_states = args.backgrounds * len(combos) * 6
    rng = np.random.default_rng(args.seed)

    prob_sum = defaultdict(float)
    offer_count = defaultdict(int)
    top_count = defaultdict(int)
    entropy_sum = 0.0

    obs_batch = []
    offered_batch = []
    done = 0
    started = time.time()

    def flush():
        nonlocal done, entropy_sum
        if not obs_batch:
            return
        probs, offs = batch_encode(model, obs_batch, offered_batch)
        for offered, p in zip(offs, probs):
            winner = int(np.argmax(p))
            entropy_sum += float(-(p * np.log(np.clip(p, 1e-12, 1.0))).sum())
            for i, relic in enumerate(offered):
                offer_count[relic] += 1
                prob_sum[relic] += float(p[i])
            top_count[offered[winner]] += 1
        done += len(offs)
        obs_batch.clear()
        offered_batch.clear()

    with torch.inference_mode():
        for _ in range(args.backgrounds):
            bg = make_background(rng)
            base_obs = make_base_obs(bg, flags)
            for triple in combos:
                for offered in permutations(triple):
                    obs_batch.append(add_boss_relics(base_obs, offered))
                    offered_batch.append(offered)
                    if len(obs_batch) >= args.batch_size:
                        flush()
        flush()

    rows = []
    for relic in BOSS_POOL:
        n = offer_count[relic]
        avg = prob_sum[relic] / n
        rows.append((
            avg,
            avg / (1.0 / 3.0),
            top_count[relic] / n,
            relic,
            n,
        ))
    rows.sort(reverse=True)

    print(f"checkpoint={ckpt}")
    print(f"device={device}")
    print(f"flags={flags}")
    print(f"backgrounds={args.backgrounds} triples={len(combos)} permutations=6 states={done}")
    print(f"mean_entropy={entropy_sum / done:.6f} uniform_entropy={math.log(3):.6f}")
    print()
    print("rank\trelic\tavg_p_if_offered\tlift_vs_uniform\targmax_rate\toffer_count")
    for idx, (avg, lift, argmax_rate, relic, n) in enumerate(rows, 1):
        print(
            f"{idx}\t{relic}\t{avg:.6f}\t{lift:.3f}\t{argmax_rate:.6f}\t{n}"
        )
    print(f"elapsed_sec={time.time() - started:.2f}")


if __name__ == "__main__":
    main()
