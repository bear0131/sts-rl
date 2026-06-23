"""Verify the shop-info bindings (gc.shop_contents / gc.legal_shop_options).

This is a BINDING check only — it deliberately does NOT touch the obs encoder
or the action codec (no OBS_VERSION / ACTION_VERSION change yet). It drives
masked-random whole runs over the thick SimGameEnv (which owns the gc), and
whenever it lands on a SHOP_ROOM it reads the shelf straight off the gc and
asserts the contract:

  - shop_contents() is POSITIONAL: cards[7], relics[3], potions[3], remove_cost;
  - a price of -1 marks a sold/empty slot;
  - legal_shop_options() returns (kind, idx) pairs, kind in
    {card,relic,potion,remove,skip}, "skip" ALWAYS present;
  - legality is consistent with the shelf: every legal buy points at a slot whose
    price is in [0, gold]; no affordable, in-stock slot is missing from legal.

Run from repo root (needs the sim built):
    python -m env.scripts.check_shop_binding --episodes 60 --seed 0
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np

from env.core.game_env import SimGameEnv


_KINDS = {"card", "relic", "potion", "remove", "skip"}


def _check_shop(gc, report):
    """Assert the shop bindings agree with each other + with gold. Mutates
    `report` counters. Returns nothing; raises AssertionError on a violation."""
    contents = gc.shop_contents()
    legal = gc.legal_shop_options()
    gold = gc.gold

    # ---- shape: strictly positional, fixed widths ----
    assert len(contents["cards"]) == 7, contents["cards"]
    assert len(contents["relics"]) == 3, contents["relics"]
    assert len(contents["potions"]) == 3, contents["potions"]
    for c in contents["cards"]:
        assert set(c) == {"id", "upgrades", "price"}, c
    for r in contents["relics"]:
        assert set(r) == {"id", "price"}, r
    for p in contents["potions"]:
        assert set(p) == {"id", "price"}, p

    # ---- legal options: (kind, idx) pairs, kinds in the closed set ----
    kinds = [k for k, _ in legal]
    for k, idx in legal:
        assert k in _KINDS, (k, idx)
        assert isinstance(idx, int) and idx >= 0, (k, idx)
    assert "skip" in kinds, f"skip must always be legal: {legal}"

    # ---- cross-check: legal buys <-> shelf prices/gold ----
    legal_set = set(legal)
    # 1) every legal buy points at an affordable, in-stock slot
    for k, idx in legal:
        if k == "card":
            price = contents["cards"][idx]["price"]
        elif k == "relic":
            price = contents["relics"][idx]["price"]
        elif k == "potion":
            price = contents["potions"][idx]["price"]
        elif k == "remove":
            price = contents["remove_cost"]
        else:  # skip
            continue
        assert price != -1, f"legal {k}#{idx} but slot is sold/empty (price -1)"
        assert 0 <= price <= gold, f"legal {k}#{idx} price {price} > gold {gold}"

    # 2) no affordable, in-stock slot is silently MISSING from legal
    for i, c in enumerate(contents["cards"]):
        if c["price"] != -1 and gold >= c["price"]:
            assert ("card", i) in legal_set, f"affordable card#{i} not legal: {c}"
    for i, r in enumerate(contents["relics"]):
        if r["price"] != -1 and gold >= r["price"]:
            assert ("relic", i) in legal_set, f"affordable relic#{i} not legal: {r}"
    for i, p in enumerate(contents["potions"]):
        if p["price"] != -1 and gold >= p["price"]:
            assert ("potion", i) in legal_set, f"affordable potion#{i} not legal: {p}"
    rc = contents["remove_cost"]
    if rc != -1 and gold >= rc:
        assert ("remove", 0) in legal_set, f"affordable remove not legal: cost {rc}"

    report["shops"] += 1
    report["buyable"] += sum(1 for k in kinds if k != "skip")
    return contents, legal


def run_episode(env, rng, report):
    env.reset(seed=int(rng.integers(0, 2**31 - 1)))
    steps = 0
    seen_shop_this_ep = False
    while not env.is_over and steps < 4000:
        if env.phase() == "shop" and not seen_shop_this_ep:
            contents, legal = _check_shop(env.gc, report)
            if report["shops"] <= 3:  # show a few real shelves
                _print_shop(env.gc.gold, contents, legal)
            seen_shop_this_ep = True
        legal = env.legal_semantic_actions()
        if not legal:
            report["stalls"] += 1
            break
        env.step(legal[int(rng.integers(0, len(legal)))])
        steps += 1


def _print_shop(gold, contents, legal):
    print(f"\n--- SHOP (gold={gold}) ---")
    for i, c in enumerate(contents["cards"]):
        up = f"+{c['upgrades']}" if c["upgrades"] else ""
        print(f"  card  {i}: {c['id']}{up:<4} price={c['price']}")
    for i, r in enumerate(contents["relics"]):
        print(f"  relic {i}: {r['id']:<24} price={r['price']}")
    for i, p in enumerate(contents["potions"]):
        print(f"  potion{i}: {p['id']:<24} price={p['price']}")
    print(f"  remove_cost={contents['remove_cost']}")
    print(f"  legal={legal}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    env = SimGameEnv()
    rng = np.random.default_rng(args.seed)
    report = {"shops": 0, "buyable": 0, "stalls": 0}
    for ep in range(args.episodes):
        run_episode(env, rng, report)

    print(f"\n==== {args.episodes} episodes ====")
    print(f"shops visited+checked : {report['shops']}")
    print(f"buyable options total : {report['buyable']}")
    print(f"event/screen stalls   : {report['stalls']}")
    assert report["shops"] > 0, "no shop reached — increase --episodes"
    print("OK: shop bindings consistent (contents <-> legality <-> gold).")


if __name__ == "__main__":
    main()
