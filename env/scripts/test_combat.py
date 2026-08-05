"""End-to-end regression test for SimCombatEnv. Requires the sim to be built
(sim/sts_lightspeed/build). Run from the repo root on the training machine:

    python -m env.scripts.test_combat

Plays N random-but-legal episodes across the act-1 pools and asserts the env
contract holds: obs shape/dtype/finiteness, the action mask agrees with what
bc.execute accepts, episodes terminate, and rewards stay finite.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from env.core.combat_env import SimCombatEnv
from env.core import actions
import slaythespire as sts


def _check_obs(env, obs):
    """v2 obs is a gymnasium Dict: validate it belongs to the env's declared
    space (shapes/dtypes/bounds), and that every scalar entry is finite."""
    space = env.observation_space
    assert space.contains(obs), "obs not contained in observation_space"
    for k, v in obs.items():
        assert np.all(np.isfinite(v)), f"obs[{k}] has non-finite values"


def _play_one(env, rng, reset_seed=None, max_steps=300):
    obs, info = env.reset(seed=reset_seed)
    _check_obs(env, obs)
    mask = info["action_mask"]
    assert mask.shape == (actions.N_ACTIONS,)
    assert mask.any(), "reset produced no legal actions"
    # A combat with no monsters never terminates (and means the encounter name
    # failed to resolve in create_battle — see encounters.py). Catch it loudly.
    n_mon = len(env.bc.to_dict()["monsters"])
    assert n_mon > 0, f"combat spawned 0 monsters (bad encounter name?)"

    steps = 0
    total_r = 0.0
    while steps < max_steps:
        legal = np.flatnonzero(mask)
        a = int(rng.choice(legal))
        obs, r, term, trunc, info = env.step(a)
        _check_obs(env, obs)
        assert np.isfinite(r), f"non-finite reward {r}"
        total_r += r
        mask = info["action_mask"]
        steps += 1
        if term or trunc:
            # On termination the last reward carries the ±1 win/loss bonus.
            # Verify its sign matches the actual outcome (guards the
            # Outcome-vs-GameOutcome enum mismatch).
            raw = env.bc.to_dict()
            outcome = raw["outcome"]  # combat Outcome: 1=victory, 2=loss
            won = outcome == 1
            if won:
                assert r > 0, f"victory but terminal reward {r} <= 0"
                assert raw["player"]["hp"] > 0, "victory but player hp <= 0"
            else:
                assert r < 0, f"loss but terminal reward {r} >= 0"
            return steps, total_r, term, won
    raise AssertionError(f"episode did not terminate in {max_steps} steps")


def _check_all_encounters_spawn():
    """Every encounter name must resolve in create_battle (>=1 monster).
    Guards against the SaveFileMappings-vs-monsterEncounterStrings name mismatch."""
    from env.core import encounters
    from env.core.factories import fixed_factory
    bad = []
    for pool, (names, _) in encounters.POOLS.items():
        for nm in names:
            env = SimCombatEnv(gc_factory=fixed_factory(seed=42, encounter=nm, floor=2))
            env.reset()
            if len(env.bc.to_dict()["monsters"]) == 0:
                bad.append((pool, nm))
    assert not bad, f"encounters spawned 0 monsters (bad names): {bad}"
    print("OK — all act-1 encounter names resolve to >=1 monster.")


def _check_wiki_card_corrections():
    def battle(card, upgraded):
        gc = sts.GameContext(sts.CharacterClass.IRONCLAD, 42, 0)
        gc.floor_num = 1
        gc.set_deck([(card, upgraded)] * 10)
        return sts.create_battle(gc, "Cultist")

    def play_first(bc):
        action = next(
            a for a in bc.legal_actions()
            if a.action_type == sts.ActionType.CARD and a.source_idx == 0
        )
        bc.execute(action)

    bc = battle("INFERNAL_BLADE", 1)
    assert all(cost == 0 for _, _, cost in bc.to_dict()["hand"])

    bc = battle("SEEING_RED", 0)
    play_first(bc)
    state = bc.to_dict()
    assert state["player"]["energy"] == 4
    assert any(card[0] == "SEEING_RED" for card in state["exhaust"])

    bc = battle("DISARM", 1)
    play_first(bc)
    powers = dict(bc.to_dict()["monsters"][0]["powers"])
    assert powers.get("Strength") == -3
    print("OK - wiki corrections: Infernal Blade+, Seeing Red, Disarm+.")


def main():
    _check_all_encounters_spawn()
    _check_wiki_card_corrections()
    rng = np.random.default_rng(0)
    n_per_pool = 5
    for pool in ("normal", "elite", "boss"):
        env = SimCombatEnv(pool=pool)
        for i in range(n_per_pool):
            # Distinct reset seed per episode for reproducibility + coverage.
            steps, total_r, term, won = _play_one(env, rng, reset_seed=1000 * (i + 1) + len(pool))
            print(f"[{pool}] episode {i}: {steps} steps, reward {total_r:+.3f}, "
                  f"won={won}, terminated={term}")
    print("OK — SimCombatEnv contract holds across act-1 pools.")


if __name__ == "__main__":
    main()
